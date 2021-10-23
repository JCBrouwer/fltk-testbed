sudo cp daemon.json /etc/docker/

sudo systemctl daemon-reload
sudo systemctl restart docker
sudo service kubelet start

sudo kubeadm reset
sudo kubeadm init
sleep 1

rm -rf $HOME/.kube/*
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
export KUBECONFIG=$HOME/.kube/config

kubectl taint nodes --all node-role.kubernetes.io/master-
sleep 5

sudo cp kube-controller-manager.yaml /etc/kubernetes/manifests/
sudo cp kube-apiserver.yaml /etc/kubernetes/manifests/
echo """evictionSoft:
  memory.available: "1000Mi"
  nodefs.available: "100Mi"
  nodefs.inodesFree: "1%"
  imagefs.available: "100Mi"
  imagefs.inodesFree: "1%"
evictionSoftGracePeriod:
  memory.available: 1m
  nodefs.available: 1m
  nodefs.inodesFree: 1m
  imagefs.available: 1m
  imagefs.inodesFree: 1m
evictionHard:
  memory.available: "100Mi"
  nodefs.available: "10Mi"
  nodefs.inodesFree: "0.1%"
  imagefs.available: "10Mi"
  imagefs.inodesFree: "0.1%"
""" | sudo tee -a /var/lib/kubelet/config.yaml
sudo service kubelet restart
sleep 5

kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml

kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.9.0/nvidia-device-plugin.yml
kubectl create -f gpushare-schd-extender.yaml

sudo cp scheduler-policy-config.json /etc/kubernetes/
sudo cp kube-scheduler.yaml /etc/kubernetes/manifests/
sleep 5

kubectl create -f device-plugin-rbac.yaml
kubectl create -f device-plugin-ds.yaml

kubectl label nodes --all gpushare=true

# helm install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard
helm install nfs-server kvaps/nfs-server-provisioner

for i in `seq 1 5`; do
kustomize build kubeflow | kubectl apply -f -
done

helm install extractor charts/extractor -f charts/fltk-values.yaml

# pip install -r requirements.txt
# docker build . --tag 192.168.1.187:6000/fltk/fltk
# docker push 192.168.1.187:6000/fltk/fltk
# sleep 3

# cd charts
# helm install flearner ./orchestrator -f fltk-values.yaml
# cd ..
# sleep 5

# echo
# kubectl -n kube-system describe secret $(kubectl -n kube-system get secret | awk '/^deployment-controller-token-/{print $1}') | awk '$1=="token:"{print $2}'
# echo

# echo 'kubectl port-forward $(kubectl get pods -l "app.kubernetes.io/name=kubernetes-dashboard,app.kubernetes.io/instance=kubernetes-dashboard" -o jsonpath="{.items[0].metadata.name}") 8443:8443 &'
# echo 'kubectl port-forward $(kubectl get pods | grep extractor | cut -d " " -f1) 6006:6006 &'