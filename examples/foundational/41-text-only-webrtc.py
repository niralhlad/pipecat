#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import io
import os
import re
import shutil
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from mcp import StdioServerParameters
from PIL import Image

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    LLMMessagesAppendFrame,
    URLImageRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    ActionResult,
    RTVIAction,
    RTVIActionArgument,
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

load_dotenv(override=True)


class UrlToImageProcessor(FrameProcessor):
    def __init__(self, aiohttp_session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self._aiohttp_session = aiohttp_session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, FunctionCallResultFrame):
            await self.push_frame(frame, direction)
            image_url = self.extract_url(frame.result)
            if image_url:
                await self.run_image_process(image_url)
                # sometimes we get multiple image urls- process 1 at a time
                await asyncio.sleep(1)
        else:
            await self.push_frame(frame, direction)

    def extract_url(self, text: str):
        pattern = r"!\[[^\]]*\]\((https?://[^)]+\.(png|jpg|jpeg|PNG|JPG|JPEG))\)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

    async def run_image_process(self, image_url: str):
        try:
            logger.debug(f"handling image from url: '{image_url}'")
            async with self._aiohttp_session.get(image_url) as response:
                image_stream = io.BytesIO(await response.content.read())
                image = Image.open(image_stream)
                image = image.convert("RGB")
                frame = URLImageRawFrame(
                    url=image_url, image=image.tobytes(), size=image.size, format="RGB"
                )
                await self.push_frame(frame)
        except Exception as e:
            error_msg = f"Error handling image url {image_url}: {str(e)}"
            logger.error(error_msg)


async def run_bot(webrtc_connection: SmallWebRTCConnection, _: argparse.Namespace):
    logger.info(f"Starting bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(),
    )

    # Create an HTTP session for API calls
    async with aiohttp.ClientSession() as session:
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        async def action_llm_append_to_messages_handler(
            rtvi: RTVIProcessor, service: str, arguments: dict[str, any]
        ) -> ActionResult:
            run_immediately = (
                arguments["run_immediately"] if "run_immediately" in arguments else True
            )

            if run_immediately:
                await rtvi.interrupt_bot()

            # We just interrupted the bot so it should be fine to use the
            # context directly instead of through frame.
            if "messages" in arguments and arguments["messages"]:
                mess = arguments["messages"]
                frame = LLMMessagesAppendFrame(messages=arguments["messages"])
                await rtvi.push_frame(frame)

            if run_immediately:
                frame = context_aggregator.user().get_context_frame()
                await rtvi.push_frame(frame)

            return True

        action_llm_append_to_messages = RTVIAction(
            service="llm",
            action="append_to_messages",
            result="bool",
            arguments=[
                RTVIActionArgument(name="messages", type="array"),
                RTVIActionArgument(name="run_immediately", type="bool"),
            ],
            handler=action_llm_append_to_messages_handler,
        )

        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
        rtvi.register_action(action_llm_append_to_messages)

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                context_aggregator.user(),
                llm,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            logger.info("Pipecat client ready.")
            await rtvi.set_bot_ready()
            # Update small webrtc UI to just handle text
            messages = {
                "show_text_container": True,
                "show_video_container": False,
                "show_debug_container": False,
            }
            rtvi_frame = RTVIServerMessageFrame(data=messages)
            await task.queue_frames([rtvi_frame])

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected: {client}")
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")

        @transport.event_handler("on_client_closed")
        async def on_client_closed(transport, client):
            logger.info(f"Client closed connection")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)


if __name__ == "__main__":
    from run import main

    main()
