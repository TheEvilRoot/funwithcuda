# ping local one 
ping -c 1 -t 1 192.168.100.4 2>&1 1>/dev/null

if [ "$?" -eq "0" ]; then
  echo "192.168.100.4"
  exit 0
fi

# ping vpn path
ping -c 1 -t 1 10.8.0.42 2>&1 1>/dev/null

if [ "$?" -eq "0" ]; then
  echo "10.8.0.42"
  exit 0
fi
echo "No address"
exit 1
