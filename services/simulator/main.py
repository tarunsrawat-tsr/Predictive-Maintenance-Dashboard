"""Simulate machine telemetry and publish it over MQTT."""

import os
import time

import certifi
import paho.mqtt.client as mqtt

from pdm.simulation import FleetSimulator, download


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default))
    return value.lower() in {"1", "true", "yes"}


def connect_mqtt() -> mqtt.Client:
    """Connect to the configured MQTT broker."""

    host = os.getenv("MQTT_HOST")

    if not host:
        raise SystemExit("MQTT_HOST is required unless DRY_RUN=true")

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=os.getenv("MQTT_CLIENT_ID", "pdm-gateway"),
    )

    if env_bool("MQTT_TLS", True):
        certificate = os.getenv("IOT_CERT_PATH")
        private_key = os.getenv("IOT_KEY_PATH")
        ca_file = os.getenv("IOT_CA_PATH", certifi.where())

        if not certificate or not private_key:
            raise SystemExit(
                "IOT_CERT_PATH and IOT_KEY_PATH are required when TLS is enabled"
            )

        client.tls_set(
            ca_certs=ca_file,
            certfile=certificate,
            keyfile=private_key,
        )

    client.connect(
        host=host,
        port=int(os.getenv("MQTT_PORT", "8883")),
        keepalive=60,
    )
    client.loop_start()

    return client


def main() -> None:
    data_dir = os.getenv("PDM_DATA_DIR", "data/cmapss")
    interval = float(os.getenv("SIM_INTERVAL", "5"))
    max_ticks = int(os.getenv("SIM_MAX_TICKS", "0")) or None
    dry_run = env_bool("DRY_RUN")

    # Download the dataset when it is not already available.
    download(data_dir)

    # Create the simulated fleet.
    simulator = FleetSimulator(
        n_machines=int(os.getenv("SIM_MACHINES", "10")),
        dataset=os.getenv("SIM_DATASET", "FD001"),
        split=os.getenv("SIM_SPLIT", "train"),
        site=os.getenv("SIM_SITE", "nagoya"),
        seed=int(os.getenv("SIM_SEED", "42")),
        noise_std=float(os.getenv("SIM_NOISE", "0")),
        data_dir=data_dir,
    )

    client = None if dry_run else connect_mqtt()
    sent = 0

    try:
        for batch in simulator.stream(max_ticks=max_ticks):
            started = time.monotonic()

            for message in batch:
                payload = message.model_dump_json()

                if dry_run:
                    print(message.topic, payload)
                else:
                    result = client.publish(
                        message.topic,
                        payload,
                        qos=1,
                    )
                    result.wait_for_publish(timeout=10)

                sent += 1

            print(f"Tick {simulator.tick}: {sent} messages sent")

            elapsed = time.monotonic() - started
            time.sleep(max(0, interval - elapsed))

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    finally:
        if client:
            client.loop_stop()
            client.disconnect()

        print(f"Stopped after sending {sent} messages.")


if __name__ == "__main__":
    main()