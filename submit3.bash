# queue_name=preempt-l20-gpu-acloud
queue_name=project-l20-saturnv-release-acloud
aidi-inf-cli job submit -f diffusiondrive.yaml -t ~/temp_dir --queue_name $queue_name &
aidi-inf-cli job submit -f diffusiondrive.yaml -t ~/temp_dir --queue_name $queue_name 
# aidi-inf-cli job submit -f diffusiondrive.yaml -t ~/temp_dir --queue_name $queue_name 
# aidi-inf-cli job submit -f s5.yaml -t ~/temp_dir --queue_name project-l20-saturnv-release-acloud

# aidi-inf-cli job submit -f diffusiondrive.yaml -t ~/temp_dir --queue_name $queue_name 

