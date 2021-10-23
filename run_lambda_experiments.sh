for lambda in 15 10 5 3; do
    for seed in 123 234 345 456 567; do
        helm uninstall flearner
        kubectl delete pytorchjob --all

        sed -i 's|"arrival_statistic": .*,|"arrival_statistic": '${lambda}',|g' configs/example_cloud_experiment.json
        sed -i 's|"arrival_seed": .*|"arrival_seed": '${seed}'|g' configs/example_cloud_experiment.json
        cat configs/example_cloud_experiment.json | grep arrival
        
        docker build . --tag 192.168.1.187:6000/fltk/fltk
        docker push 192.168.1.187:6000/fltk/fltk

        until [ -z $(kubectl get pods --all-namespaces | grep "fl-server") ]; do sleep 1; done

        helm install flearner charts/orchestrator -f charts/fltk-values.yaml

        sleep $((12 * 60 + 10))

        kubectl logs fl-server > lambda_results/random_${lambda}_${seed}.log
    done
done