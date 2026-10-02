#!/usr/bin/env python3
"""
scripts/deploy_polygon.py
-------------------------
Deploys PatientRecordRegistry.sol to Polygon Amoy Testnet (Chain ID 80002).
"""

import json
import os
import sys
from pathlib import Path
from eth_account import Account
from web3 import Web3


def main():
    base_dir = Path(__file__).resolve().parents[1]
    artifact_path = base_dir / "contracts" / "PatientRecordRegistry.json"

    if not artifact_path.exists():
        print(f"Error: {artifact_path} not found. Please compile the contract first.")
        sys.exit(1)

    artifact = json.loads(artifact_path.read_text())
    abi = artifact["abi"]
    bytecode = artifact["bytecode"]

    rpc_url = os.getenv("POLYGON_RPC_URL", "https://rpc-amoy.polygon.technology/")
    private_key = os.getenv("POLYGON_PRIVATE_KEY")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print(f"Error: Unable to connect to Polygon RPC at {rpc_url}")
        sys.exit(1)

    print("==================================================")
    print("      BioNexus Polygon Smart Contract Deployer    ")
    print("==================================================")
    print(f"Network:  Polygon Amoy Testnet (Chain ID: 80002)")
    print(f"RPC Node: {rpc_url}")

    if not private_key:
        print("\nNo POLYGON_PRIVATE_KEY environment variable found.")
        print("Please provide a private key funded with testnet POL.")
        print("(Get free testnet POL from: https://faucet.polygon.technology/)\n")
        private_key = input("Enter your Polygon private key (or press Ctrl+C to cancel): ").strip()
        if not private_key:
            print("Deployment cancelled.")
            sys.exit(1)

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    account = Account.from_key(private_key)
    balance_wei = w3.eth.get_balance(account.address)
    balance_pol = w3.from_wei(balance_wei, "ether")

    print(f"\nDeployer Address: {account.address}")
    print(f"Current Balance:  {balance_pol:.4f} POL")

    if balance_wei == 0:
        print("\n❌ Error: Your wallet balance is 0 POL.")
        print("You need a small amount of testnet POL for gas (free from faucet).")
        print(f"1. Go to: https://faucet.polygon.technology/")
        print(f"2. Select 'Amoy' network")
        print(f"3. Paste your address: {account.address}")
        print("4. Re-run this script after receiving tokens.")
        sys.exit(1)

    print("\nDeploying PatientRecordRegistry contract to Polygon Amoy...")
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    nonce = w3.eth.get_transaction_count(account.address)
    gas_price = w3.eth.gas_price

    tx = contract.constructor().build_transaction({
        "chainId": 80002,
        "from": account.address,
        "nonce": nonce,
        "gasPrice": gas_price,
    })

    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"Transaction Broadcasted! Tx Hash: {w3.to_hex(tx_hash)}")
    print(f"Tracking Tx on PolygonScan: https://amoy.polygonscan.com/tx/{w3.to_hex(tx_hash)}")
    print("Waiting for block confirmation (~2 to 5 seconds)...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    contract_address = receipt.contractAddress

    print("\n✅ Contract Successfully Deployed!")
    print(f"Contract Address: {contract_address}")
    print(f"PolygonScan Link: https://amoy.polygonscan.com/address/{contract_address}")

    # Save to local config/.env
    env_file = base_dir / ".env"
    env_lines = []
    if env_file.exists():
        env_lines = [l for l in env_file.read_text().splitlines() if not l.startswith("POLYGON_")]

    env_lines.append(f'POLYGON_RPC_URL="{rpc_url}"')
    env_lines.append(f'POLYGON_PRIVATE_KEY="{private_key}"')
    env_lines.append(f'POLYGON_CONTRACT_ADDRESS="{contract_address}"')
    env_file.write_text("\n".join(env_lines) + "\n")

    print("\nUpdated .env with your live Polygon contract settings.")
    print("Restart your FastAPI backend server to serve live Polygon transactions!")


if __name__ == "__main__":
    main()
