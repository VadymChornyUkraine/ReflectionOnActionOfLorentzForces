while [ 1 ]; do sleep 10; ping -c 1 google.com; if [ $? = 0 ]; then echo "it is good"; else nmcli radio wifi off; nmcli radio wifi on; sleep 10; nmcli d wifi connect WiFi_Free; fi; done
