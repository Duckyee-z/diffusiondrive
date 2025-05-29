# queue_name=preempt-l20-gpu-acloud
queue_name=share-4090-small-idc-newage
aidi-inf-cli job submit -f diffusiondrive.yaml -t ~/temp_dir --queue_name $queue_name &
aidi-inf-cli job submit -f diffusiondrive.yaml -t ~/temp_dir --queue_name $queue_name &
aidi-inf-cli job submit -f diffusiondrive.yaml -t ~/temp_dir --queue_name $queue_name

