helm uninstall flearner
kubectl delete pytorchjob --all

#for lambda in 10 5 4 3; do
#    for schedule in vram-aware random; do
#        for seed in 22 03 1996; do
#            until ! kubectl describe nodes | grep "MemoryPressure       True"; do
#                sleep 1
#                kubectl taint nodes --all node.kubernetes.io/memory-pressure-
#            done

#            sed -i 's|"arrival_statistic": .*,|"arrival_statistic": '${lambda}',|g' configs/example_cloud_experiment.json
#            sed -i 's|"scheduling": .*,|"scheduling": "'${schedule}'",|g' configs/example_cloud_experiment.json
#            sed -i 's|"arrival_seed": .*|"arrival_seed": '${seed}'|g' configs/example_cloud_experiment.json
#            cat configs/example_cloud_experiment.json | grep "arrival\|scheduling"
            
#            docker build . --tag 192.168.1.187:6000/fltk/fltk
#            docker push 192.168.1.187:6000/fltk/fltk
            
#            helm uninstall flearner
#            kubectl delete pytorchjob --all

#            until [ -z $(kubectl get pods --all-namespaces | grep "fl-server") ]; do
#                sleep 1
#            done

#            {
#                timeout $((10 * 60 + 35)) nvidia-smi \
#                --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
#                --format=csv -l 1 > lambda_results/${schedule}_${lambda}_${seed}.gpustats
#            } &

#            helm install flearner charts/orchestrator -f charts/fltk-values.yaml

#            sleep $((10 * 60 + 30))

#            kubectl logs fl-server > lambda_results/${schedule}_${lambda}_${seed}.log
#            kubectl get pods | head -n -3 | tail -n +2 | awk '{print $4}' | paste -sd+ | bc > lambda_results/${schedule}_${lambda}_${seed}.restarts
#            kubectl get pods | grep Completed | wc -l > lambda_results/${schedule}_${lambda}_${seed}.completed
#            kubectl get pods | grep Evicted | wc -l > lambda_results/${schedule}_${lambda}_${seed}.evicted

#            helm uninstall flearner
#            kubectl delete pytorchjob --all
#        done
#    done
#done

#git add .
#git stash
#git checkout 31c376b
#git merge --squash --strategy-option=theirs stash

for lambda in 10 5 4 3 2 1; do
    for schedule in improved; do
        for seed in 22 03 1996; do
            until ! kubectl describe nodes | grep "MemoryPressure       True"; do
                sleep 1
                kubectl taint nodes --all node.kubernetes.io/memory-pressure-
            done

            sed -i 's|"arrival_statistic": .*,|"arrival_statistic": '${lambda}',|g' configs/example_cloud_experiment.json
            sed -i 's|"scheduling": .*,|"scheduling": "'${schedule}'",|g' configs/example_cloud_experiment.json
            sed -i 's|"arrival_seed": .*|"arrival_seed": '${seed}'|g' configs/example_cloud_experiment.json
            cat configs/example_cloud_experiment.json | grep "arrival\|scheduling"

            docker build . --tag 192.168.1.187:6000/fltk/fltk
            docker push 192.168.1.187:6000/fltk/fltk

            helm uninstall flearner
            kubectl delete pytorchjob --all

            until [ -z $(kubectl get pods --all-namespaces | grep "fl-server") ]; do
                sleep 1
            done

            {
                timeout $((10 * 60 + 35)) nvidia-smi \
                --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
                --format=csv -l 1 > lambda_results/${schedule}_${lambda}_${seed}.gpustats
            } &

            helm install flearner charts/orchestrator -f charts/fltk-values.yaml

            sleep $((10 * 60 + 30))

            kubectl logs fl-server > lambda_results/${schedule}_${lambda}_${seed}.log
            kubectl logs fl-server | grep "restarts=" | cut -d= -f2 > lambda_results/${schedule}_${lambda}_${seed}.restarts
            kubectl get pods | grep Completed | wc -l > lambda_results/${schedule}_${lambda}_${seed}.completed
            kubectl get pods | grep Evicted | wc -l > lambda_results/${schedule}_${lambda}_${seed}.evicted

            helm uninstall flearner
            kubectl delete pytorchjob --all
        done
    done
done
