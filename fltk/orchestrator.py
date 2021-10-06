from datetime import datetime
import logging
import random
import time
import uuid
from queue import PriorityQueue
from typing import Dict, List

from kubeflow.pytorchjob import PyTorchJobClient
from kubeflow.pytorchjob.constants.constants import PYTORCHJOB_GROUP, PYTORCHJOB_PLURAL, PYTORCHJOB_VERSION
from kubeflow.pytorchjob.models.v1_py_torch_job import V1PyTorchJob

from fltk.util.cluster.client import ClusterManager, construct_inference_job
from fltk.util.config.base_config import BareConfig
from fltk.util.task.generator.arrival_generator import Arrival, ArrivalGenerator
from fltk.util.task.task import ArrivalTask


class Orchestrator(object):
    """
    Central component of the Federated Learning System: The Orchestrator

    The Orchestrator is in charge of the following tasks:
    - Running experiments
        - Creating and/or managing tasks
        - Keep track of progress (pending/started/failed/completed)
    - Keep track of timing

    Note that the Orchestrator does not function like a Federator, in the sense that it keeps a central model, performs
    aggregations and keeps track of Clients. For this, the KubeFlow PyTorch-Operator is used to deploy a train task as
    a V1PyTorchJob, which automatically generates the required setup in the cluster. In addition, this allows more Jobs
    to be scheduled, than that there are resources, as such, letting the Kubernetes Scheduler let decide when to run
    which containers where.
    """

    _alive = False
    # Priority queue, requires an orderable object, otherwise a Tuple[int, Any] can be used to insert.
    pending_tasks: "PriorityQueue[ArrivalTask]" = PriorityQueue()

    deployed_jobs: Dict[uuid.UUID, V1PyTorchJob] = {}

    arrival_times: Dict[uuid.UUID, datetime] = {}
    task_info: Dict = {}

    def __init__(self, cluster_mgr: ClusterManager, arv_gen: ArrivalGenerator, config: BareConfig):
        self.__logger = logging.getLogger("Orchestrator")
        self.__logger.debug("Loading in-cluster configuration")
        self.__cluster_mgr = cluster_mgr
        self.__arrival_generator = arv_gen
        self._config = config

        # API to interact with the cluster.
        self.__client = PyTorchJobClient()

    def stop(self) -> None:
        """
        Stop the Orchestrator.
        @return:
        @rtype:
        """
        self.__logger.info("Received stop signal for the Orchestrator. Waiting 1 minute for jobs to finish")
        time.sleep(120)
        times = self.__cluster_mgr.get_finish_times()
        print("\n\n\n")
        print("id,network,job_type,image_size,num_imgs,device,batch_size,data_parallelism,response_time")
        for id, arrival in self.arrival_times.items():
            network, image_size, job_type, num_imgs, device, batch_size, data_parallelism = self.task_info[id]
            if str(id) in times:
                _, finish = times[str(id)]
                response_time = int((finish.replace(tzinfo=None) - arrival.replace(tzinfo=None)).total_seconds() * 1000)
            else:
                response_time = -1
            print(
                ",".join(
                    [
                        str(id).split("-")[0],
                        f"{network}",
                        f"{job_type}",
                        f"{image_size}",
                        f"{num_imgs}",
                        f"{device}",
                        f"{batch_size}",
                        f"{data_parallelism}",
                        f"{response_time}",
                    ]
                )
            )
        print("\n\n\n")
        self.__cluster_mgr.stop()
        self._alive = False
        exit(0)

    def run(self, clear: bool = True) -> None:
        """
        Main loop of the Orchestartor.
        @param clear: Boolean indicating whether a previous deployment needs to be cleaned up (i.e. lingering jobs that
        were deployed by the previous run).

        @type clear: bool
        @return: None
        @rtype: None
        """
        self._alive = True
        start_time = time.time()
        if clear:
            self.__clear_jobs()
        while self._alive and time.time() - start_time < self._config.get_duration():
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            while not self.__arrival_generator.arrivals.empty():
                arrival: Arrival = self.__arrival_generator.arrivals.get()
                unique_identifier: uuid.UUID = uuid.uuid4()
                task = ArrivalTask(
                    priority=arrival.get_priority(),
                    id=unique_identifier,
                    network=arrival.get_network(),
                    dataset=arrival.get_dataset(),
                    sys_conf=arrival.get_system_config(),
                    param_conf=arrival.get_parameter_config(),
                )

                self.__logger.debug(f"Arrival of: {task}")
                self.pending_tasks.put(task)
                self.arrival_times[task.id] = datetime.now()

            while not self.pending_tasks.empty():
                # Do blocking request to priority queue
                curr_task = self.pending_tasks.get()

                # edit curr_task to schedule to a device with some batch size and parallelism
                curr_task.param_conf.device = (
                    f"cuda:{random.choices([0, 1], [2/3, 1/3])[0]}"  # or "cuda:0", "cuda:1", etc.
                )
                curr_task.param_conf.bs = 8
                curr_task.sys_conf.data_parallelism = 1

                self.task_info[task.id] = (
                    curr_task.network,
                    curr_task.param_conf.image_size,
                    curr_task.param_conf.job_type,
                    curr_task.param_conf.num_imgs,
                    curr_task.param_conf.device,
                    curr_task.param_conf.bs,
                    curr_task.sys_conf.data_parallelism,
                )

                # random scheduling

                # greedy scheduling
                #   VRAM check
                #   best device with space

                # gavel scheduling

                self.__logger.info(f"Scheduling arrival of Arrival: {curr_task.id}")
                job_to_start = construct_inference_job(self._config, curr_task)

                # Hack to overcome limitation of KubeFlow version (Made for older version of Kubernetes)
                self.__logger.info(f"Deploying on cluster: {curr_task.id}")
                self.__client.create(job_to_start, namespace=self._config.cluster_config.namespace)
                self.deployed_jobs[curr_task.id] = job_to_start

            time.sleep(5)

        self.stop()

    def __clear_jobs(self):
        """
        Function to clear existing jobs in the environment (i.e. old experiments/tests)
        @return: None
        @rtype: None
        """
        namespace = self._config.cluster_config.namespace
        self.__logger.info(f"Clearing old jobs in current namespace: {namespace}")

        for job in self.__client.get(namespace=self._config.cluster_config.namespace)["items"]:
            job_name = job["metadata"]["name"]
            self.__logger.info(f"Deleting: {job_name}")
            try:
                self.__client.custom_api.delete_namespaced_custom_object(
                    PYTORCHJOB_GROUP, PYTORCHJOB_VERSION, namespace, PYTORCHJOB_PLURAL, job_name
                )
            except Exception as e:
                self.__logger.warning(f"Could not delete: {job_name}")
                print(e)
