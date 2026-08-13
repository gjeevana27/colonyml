import socket
import threading
import time
import os
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, ServiceStateChange
from typing import List, Dict, Callable


class NodeDiscovery:
    """
    Automatically finds other ColonyML nodes on the same network.
    Uses mDNS/Zeroconf — same technology as AirDrop.

    Usage:
        discovery = NodeDiscovery(port=29500)
        discovery.announce()
        discovery.start_listening()
        nodes = discovery.get_all_nodes()
    """

    SERVICE_TYPE = "_colonyml._tcp.local."

    def __init__(self, port: int = 29500):
        self.port = port
        self.hostname = socket.gethostname()
        self.local_ip = self._get_local_ip()
        self.zeroconf = Zeroconf()
        self.discovered_nodes: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._on_node_found: List[Callable] = []
        self._on_node_lost: List[Callable] = []

    def _get_local_ip(self) -> str:
        """Get this machine's local IP address."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()

    def _get_cpu_cores(self) -> int:
        return os.cpu_count() or 1

    def announce(self):
        """
        Announce this node to the network.
        Other machines will automatically discover us.
        """
        info = ServiceInfo(
            self.SERVICE_TYPE,
            f"{self.hostname}.{self.SERVICE_TYPE}",
            addresses=[socket.inet_aton(self.local_ip)],
            port=self.port,
            properties={
                b"hostname": self.hostname.encode(),
                b"cores": str(self._get_cpu_cores()).encode(),
                b"version": b"0.1.1"
            }
        )
        self.zeroconf.register_service(info)
        print(f"[ColonyML] Node announced: {self.hostname} "
              f"({self.local_ip}:{self.port}) "
              f"| {self._get_cpu_cores()} cores")

    def start_listening(self):
        """Listen for other nodes on the network."""
        ServiceBrowser(
            self.zeroconf,
            self.SERVICE_TYPE,
            handlers=[self._on_service_state_change]
        )
        print("[ColonyML] Listening for other nodes...")

    def _on_service_state_change(
        self,
        zeroconf,
        service_type,
        name,
        state_change
    ):
        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info and info.addresses:
                ip = socket.inet_ntoa(info.addresses[0])
                props = info.properties or {}
                node_name = props.get(
                    b"hostname", b"unknown"
                ).decode()
                cores = int(
                    props.get(b"cores", b"1").decode()
                )

                # Don't add ourselves
                if ip != self.local_ip:
                    with self._lock:
                        self.discovered_nodes[ip] = {
                            "ip": ip,
                            "port": info.port,
                            "hostname": node_name,
                            "cores": cores
                        }
                    print(
                        f"[ColonyML] Found node: "
                        f"{node_name} ({ip}) "
                        f"| {cores} cores"
                    )
                    for callback in self._on_node_found:
                        callback(self.discovered_nodes[ip])

        elif state_change is ServiceStateChange.Removed:
            info = zeroconf.get_service_info(service_type, name)
            if info and info.addresses:
                ip = socket.inet_ntoa(info.addresses[0])
                with self._lock:
                    if ip in self.discovered_nodes:
                        node = self.discovered_nodes.pop(ip)
                        print(
                            f"[ColonyML] Lost node: "
                            f"{node['hostname']}"
                        )
                        for callback in self._on_node_lost:
                            callback(node)

    def get_all_nodes(self) -> List[dict]:
        """Get all nodes including this machine."""
        with self._lock:
            nodes = list(self.discovered_nodes.values())

        nodes.append({
            "ip": self.local_ip,
            "port": self.port,
            "hostname": self.hostname,
            "cores": self._get_cpu_cores(),
            "is_self": True
        })

        return sorted(nodes, key=lambda x: x["ip"])

    def on_node_found(self, callback: Callable):
        """Register callback when a new node is found."""
        self._on_node_found.append(callback)

    def on_node_lost(self, callback: Callable):
        """Register callback when a node leaves."""
        self._on_node_lost.append(callback)

    def stop(self):
        """Shut down discovery."""
        self.zeroconf.unregister_all_services()
        self.zeroconf.close()
        print("[ColonyML] Discovery stopped.")


if __name__ == "__main__":
    discovery = NodeDiscovery(port=29500)
    discovery.announce()
    discovery.start_listening()

    print("\nWaiting for nodes... (Ctrl+C to stop)\n")
    try:
        while True:
            time.sleep(3)
            nodes = discovery.get_all_nodes()
            print(f"Cluster: {len(nodes)} node(s)")
            for node in nodes:
                tag = "(you)" if node.get("is_self") else ""
                print(
                    f"  {node['hostname']} "
                    f"({node['ip']}) "
                    f"| {node['cores']} cores {tag}"
                )
    except KeyboardInterrupt:
        discovery.stop()