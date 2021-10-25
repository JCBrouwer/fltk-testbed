helm uninstall flearner
kubectl delete pytorchjob --all

for lambda in 10 5 4 3; do
    for schedule in vram-aware random; do
        for seed in 22 03 1996; do
            echo until ! kubectl describe nodes | grep "MemoryPressure       True"
            echo    do sleep 1
            echo    kubectl taint nodes --all node.kubernetes.io/memory-pressure-
            echo done
echo 
            echo sed -i 's|"arrival_statistic": .*,|"arrival_statistic": '${lambda}',|g' configs/example_cloud_experiment.json
            echo sed -i 's|"scheduling": .*,|"scheduling": "'${schedule}'",|g' configs/example_cloud_experiment.json
            echo sed -i 's|"arrival_seed": .*|"arrival_seed": '${seed}'|g' configs/example_cloud_experiment.json
            echo cat configs/example_cloud_experiment.json | grep "arrival\|scheduling"
            echo 
            echo docker build . --tag 192.168.1.187:6000/fltk/fltk
            echo docker push 192.168.1.187:6000/fltk/fltk
            echo 
            echo helm uninstall flearner
            echo kubectl delete pytorchjob --all
echo 
            echo until [ -z $(kubectl get pods --all-namespaces | grep "fl-server") ]
            echo    do sleep 1
            echo done
echo 
            echo {
            echo     timeout $((10 * 60 + 35)) nvidia-smi \
            echo     --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
            echo     --format=csv -l 1 > lambda_results/${schedule}_${lambda}_${seed}.gpustats
            echo } &
echo 
            echo helm install flearner charts/orchestrator -f charts/fltk-values.yaml
echo 
            echo sleep $((10 * 60 + 30))
echo 
            echo kubectl logs fl-server > lambda_results/${schedule}_${lambda}_${seed}.log
            echo kubectl get pods | head -n -3 | tail -n +2 | awk '{print $4}' | paste -sd+ | bc > lambda_results/${schedule}_${lambda}_${seed}.restarts
            echo kubectl get pods | grep Completed | wc -l > lambda_results/${schedule}_${lambda}_${seed}.completed
            echo kubectl get pods | grep Evicted | wc -l > lambda_results/${schedule}_${lambda}_${seed}.evicted
echo 
            echo helm uninstall flearner
            echo kubectl delete pytorchjob --all
        done
    done
done

git checkout hans

for lambda in 10 5 4 3 2 1; do
    for schedule in improved; do
        for seed in 22 03 1996; do
            echo until ! kubectl describe nodes | grep "MemoryPressure       True"
            echo    do sleep 1
            echo    kubectl taint nodes --all node.kubernetes.io/memory-pressure-
            echo doneecho 
            echo sed -i 's|"arrival_statistic": .*,|"arrival_statistic": '${lambda}',|g' configs/example_cloud_experiment.json
            echo sed -i 's|"scheduling": .*,|"scheduling": "'${schedule}'",|g' configs/example_cloud_experiment.json
            echo sed -i 's|"arrival_seed": .*|"arrival_seed": '${seed}'|g' configs/example_cloud_experiment.json
            echo cat configs/example_cloud_experiment.json | grep "arrival\|scheduling"
            echo 
            echo docker build . --tag 192.168.1.187:6000/fltk/fltk
            echo docker push 192.168.1.187:6000/fltk/fltk
            echo 
            echo helm uninstall flearner
            echo kubectl delete pytorchjob --all
echo 
            echo until [ -z $(kubectl get pods --all-namespaces | grep "fl-server") ]
            echo    do sleep 1
            echo doneecho 
            echo {
            echo     timeout $((10 * 60 + 35)) nvidia-smi \
            echo     --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
            echo     --format=csv -l 1 > lambda_results/${schedule}_${lambda}_${seed}.gpustats
            echo } &
echo 
            echo helm install flearner charts/orchestrator -f charts/fltk-values.yaml
echo 
            echo sleep $((10 * 60 + 30))
echo 
            echo kubectl logs fl-server > lambda_results/${schedule}_${lambda}_${seed}.log
            echo kubectl logs fl-server | grep "restarts=" | cut -d= -f2 > lambda_results/${schedule}_${lambda}_${seed}.restarts
            echo kubectl get pods | grep Completed | wc -l > lambda_results/${schedule}_${lambda}_${seed}.completed
            echo kubectl get pods | grep Evicted | wc -l > lambda_results/${schedule}_${lambda}_${seed}.evicted
echo 
            echo helm uninstall flearner
            echo kubectl delete pytorchjob --all
        done
    done
done