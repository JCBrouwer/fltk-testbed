sudo service docker start
sudo service kubelet start
sudo service docker restart
sudo service kubelet restart

kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.9.0/nvidia-device-plugin.yml
kubectl create -f gpushare-schd-extender.yaml
kubectl create -f device-plugin-rbac.yaml
kubectl create -f device-plugin-ds.yaml

kubectl label node ubunu94025 gpushare=true

helm install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard
helm install nfs-server kvaps/nfs-server-provisioner --set persistence.enabled=true,persistence.storageClass=standard,persistence.size=20Gi

for i in `seq 1 5`; do
kustomize build kubeflow | kubectl apply -f -
done

pip install -r requirements.txt
DOCKER_BUILDKIT=1 docker build . --tag 192.168.1.187:6000/fltk/fltk
docker push 192.168.1.187:6000/fltk/fltk
sleep 3

cd charts
helm install extractor ./extractor -f fltk-values.yaml
helm install flearner ./orchestrator -f fltk-values.yaml
cd ..
sleep 5

export DASHBOARD=$(kubectl get pods -l "app.kubernetes.io/name=kubernetes-dashboard,app.kubernetes.io/instance=kubernetes-dashboard" -o jsonpath="{.items[0].metadata.name}")
kubectl -n kube-system describe secret $(kubectl -n kube-system get secret | awk '/^deployment-controller-token-/{print $1}') | awk '$1=="token:"{print $2}'

kubectl port-forward $DASHBOARD 8443:8443 &
kubectl port-forward fl-extractor 6006:6006 &