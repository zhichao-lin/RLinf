import sys
import os
import asyncio
import torch
from rlinf.scheduler import (
    Worker,
    Cluster,
    PackedPlacementStrategy,
    Channel,
)

class Producer(Worker):
    def __init__(self):
        super().__init__()

    def produce(self, channel: Channel):
        # Synchronous put of common object
        channel.put("Hello from Producer")

        # Synchronous put of tensor
        tensor = torch.ones(1, device=self.torch_platform.current_device())
        print(f"producer {tensor=}, {tensor.device=}")
        channel.put(tensor)

        # Asynchronous put of common object
        async_work = channel.put(
            "Hello from Producer asynchronously", async_op=True
        )
        async_work.wait()

        # Asynchronous put using asyncio
        async_work = channel.put(tensor, async_op=True)

        async def wait_async():
            await async_work.async_wait()

        asyncio.run(wait_async())

        # Put object with weight
        channel.put("Hello with weight", weight=1)
        channel.put(tensor, weight=2)

class Consumer(Worker):
    def __init__(self):
        super().__init__()

    def consume(self, channel: Channel):
        print(channel.get())

        tensor = channel.get()
        print(f"consumer {tensor=}, {tensor.device=}")

        async_work = channel.get(async_op=True)
        async_result = async_work.wait()
        print(f"{async_result=}")

        async_work = channel.get(async_op=True)
        async def wait_async():
            result = await async_work.async_wait()
            print(f"{result=}, {result.device=}")

        asyncio.run(wait_async())

        # Get batch of objects based on weight
        batch = channel.get_batch(target_weight=3)
        for sample in batch:
            print(f"{sample=}")


cluster = Cluster(num_nodes=1)
channel = Channel.create(name="channel")
placement = PackedPlacementStrategy(
    start_hardware_rank=0, end_hardware_rank=0
)
producer = Producer.create_group().launch(
    cluster, name="test", placement_strategy=placement
)
placement = PackedPlacementStrategy(
    start_hardware_rank=1, end_hardware_rank=1
)
consumer = Consumer.create_group().launch(
    cluster, name="test2", placement_strategy=placement
)
r1 = producer.produce(channel)
r2 = consumer.consume(channel)
res = r1.wait()
res = r2.wait()

# producer -> channel -> consumer
# ChannelWorker uses GPU:0 of the node by default: Worker.torch_platform.current_device()
