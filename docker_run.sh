docker run --gpus all --name my_torch_flask_project --shm-size 8G -it -p 8080:8080 -p 5000:5000 -v $PWD/:/src my_torch_flask_project bash


