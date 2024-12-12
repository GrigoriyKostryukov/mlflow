# Dockerfile
```sudo usermod -aG docker $USER
and reboot

ls -l build_docker.sh # Check permissions
chmod +x build_docker.sh # Grant execute permission
./build_docker.sh    # Run the script

./run_docker.sh
```
# In docker
```
mlflow server --host 127.0.0.1 --port 5000
jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root
```
## Для второго входа в контейнер можете использовать docker exec

# Lints
`
make lint/install lint lint/fix
`

# Зависимости
auto req.txt

pip install pipreqs
pipreqs src


# Install
```
pip install .
from mypackage import 
```
# Пример запуска
`
root@d0f3bfcd03f1:/src# python3 src
Start ...
`
# Задание
По аналогии сделайте для ноутбукв
Add `flake8 src/`
