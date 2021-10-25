helm uninstall flearner
kubectl delete pytorchjob --all

for lambda in 4 3; do # 15 10 5 3
    for schedule in vram-aware random; do
        for seed in 123 234 345 456 567; do
            sed -i 's|"arrival_statistic": .*,|"arrival_statistic": '${lambda}',|g' configs/example_cloud_experiment.json
            sed -i 's|"scheduling": .*,|"scheduling": "'${schedule}'",|g' configs/example_cloud_experiment.json
            sed -i 's|"arrival_seed": .*|"arrival_seed": '${seed}'|g' configs/example_cloud_experiment.json
            cat configs/example_cloud_experiment.json | grep "arrival\|scheduling"
            
            docker build . --tag 192.168.1.187:6000/fltk/fltk
            docker push 192.168.1.187:6000/fltk/fltk

            until [ -z $(kubectl get pods --all-namespaces | grep "fl-server") ]; do sleep 1; done

            helm install flearner charts/orchestrator -f charts/fltk-values.yaml

            sleep $((10 * 60 + 20))

            kubectl logs fl-server > lambda_results/${schedule}_${lambda}_${seed}.log
            kubectl get pods | head -n -3 | tail -n +2 | awk '{print $4}' | paste -sd+ | bc > lambda_results/${schedule}_${lambda}_${seed}.restarts
            kubectl get pods | grep Completed | wc -l > lambda_results/${schedule}_${lambda}_${seed}.completed
            kubectl get pods | grep Evicted | wc -l > lambda_results/${schedule}_${lambda}_${seed}.evicted

            helm uninstall flearner
            kubectl delete pytorchjob --all
        done
    done
done