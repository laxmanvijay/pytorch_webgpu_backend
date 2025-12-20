## Design of pytorch inc extension

* The pytorch_inc_extension repository contains the code for creating a pytorch distributed plugin.
* This package depends on another static library pytorch_inc_compute which performs the network compute.
* The pytorch_inc_compute creates socket, constructs the udp packet and sends it to the switch.

* There is a folder within pytorch_inc_compute called simulated_switch which is a python socket server that simulates compute using a switch.

* In the actual flow, the packet will be intercepted by DPDK, which will bypass the kernel and send to the switch, receive the response and send it back to the pytorch_inc_compute.