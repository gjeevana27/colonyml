import pytest
from colonyml.scheduler import NodeScheduler


def make_node(ip, hostname, cores, cpu_usage, freq, ram_avail, ram_total):
    return {
        "ip": ip,
        "hostname": hostname,
        "metrics": {
            "cores": cores,
            "cpu_usage": cpu_usage,
            "cpu_freq_mhz": freq,
            "ram_available_gb": ram_avail,
            "ram_total_gb": ram_total
        }
    }


def test_batch_sums_to_total():
    scheduler = NodeScheduler()
    nodes = [
        make_node("192.168.1.1", "node1", 8, 20, 3000, 8, 16),
        make_node("192.168.1.2", "node2", 4, 50, 2000, 4, 8)
    ]
    assignments = scheduler.assign_batch_sizes(nodes, 256)
    assert sum(assignments.values()) == 256


def test_faster_node_gets_more_work():
    scheduler = NodeScheduler()
    nodes = [
        make_node("192.168.1.1", "fast", 16, 10, 4000, 16, 32),
        make_node("192.168.1.2", "slow", 2, 80, 1500, 1, 4)
    ]
    assignments = scheduler.assign_batch_sizes(nodes, 256)
    assert assignments["192.168.1.1"] > assignments["192.168.1.2"]


def test_single_node_gets_all_work():
    scheduler = NodeScheduler()
    nodes = [make_node("192.168.1.1", "solo", 8, 20, 3000, 8, 16)]
    assignments = scheduler.assign_batch_sizes(nodes, 256)
    assert assignments["192.168.1.1"] == 256


def test_empty_nodes():
    scheduler = NodeScheduler()
    assignments = scheduler.assign_batch_sizes([], 256)
    assert assignments == {}


def test_busy_node_gets_less_work():
    scheduler = NodeScheduler()
    nodes = [
        make_node("192.168.1.1", "idle", 8, 5, 3000, 8, 16),
        make_node("192.168.1.2", "busy", 8, 90, 3000, 8, 16)
    ]
    assignments = scheduler.assign_batch_sizes(nodes, 256)
    assert assignments["192.168.1.1"] > assignments["192.168.1.2"]


def test_batch_size_minimum_one():
    scheduler = NodeScheduler()
    nodes = [
        make_node("192.168.1.1", "strong", 16, 5, 4000, 16, 32),
        make_node("192.168.1.2", "weak", 1, 99, 800, 0.5, 2)
    ]
    assignments = scheduler.assign_batch_sizes(nodes, 10)
    for batch in assignments.values():
        assert batch >= 1


def test_three_nodes_sum_correct():
    scheduler = NodeScheduler()
    nodes = [
        make_node("192.168.1.1", "node1", 16, 20, 3500, 14, 16),
        make_node("192.168.1.2", "node2", 8, 40, 2500, 6, 8),
        make_node("192.168.1.3", "node3", 4, 70, 1800, 2, 8)
    ]
    assignments = scheduler.assign_batch_sizes(nodes, 512)
    assert sum(assignments.values()) == 512


def test_local_metrics_returns_dict():
    scheduler = NodeScheduler()
    metrics = scheduler.get_local_metrics()
    assert "cores" in metrics
    assert "cpu_usage" in metrics
    assert "cpu_freq_mhz" in metrics
    assert "ram_available_gb" in metrics
    assert "ram_total_gb" in metrics


def test_compute_weight_positive():
    scheduler = NodeScheduler()
    metrics = {
        "cores": 8,
        "cpu_usage": 30.0,
        "cpu_freq_mhz": 3000,
        "ram_available_gb": 8.0,
        "ram_total_gb": 16.0
    }
    weight = scheduler.compute_node_weight(metrics)
    assert weight > 0