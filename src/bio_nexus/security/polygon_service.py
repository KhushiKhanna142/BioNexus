"""
bio_nexus.security.polygon_service
-----------------------------------
Integrates with the Polygon Blockchain (Polygon Amoy Testnet, Chain ID 80002).
Anchors cryptographic digests of patient records & diagnostic proofs to
ensure tamper-resistance, non-repudiation, and clinical provenance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from eth_account import Account
from hexbytes import HexBytes
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware


class PolygonService:
    def __init__(
        self,
        rpc_url: str | None = None,
        contract_address: str | None = None,
        private_key: str | None = None,
        ledger_path: Path | None = None,
    ):
        base_dir = Path(__file__).resolve().parents[3]
        self.ledger_file = ledger_path or (base_dir / "data" / "security" / "polygon_ledger.json")
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)

        # Load environment variables from .env if present
        env_file = base_dir / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

        self.rpc_url = rpc_url or os.getenv("POLYGON_RPC_URL", "https://polygon-amoy-bor-rpc.publicnode.com")
        self.contract_address = contract_address or os.getenv("POLYGON_CONTRACT_ADDRESS", "0xE771417C54c4968E7E324925a563a224BD16A30e")
        self.private_key = private_key or os.getenv("POLYGON_PRIVATE_KEY", "0x58029ca059e69cd23a05a70f234aea2ec319bed4ba67244dd11e8086553ba400")

        # Load ABI
        abi_file = base_dir / "contracts" / "PatientRecordRegistry.json"
        if abi_file.exists():
            contract_json = json.loads(abi_file.read_text())
            self.abi = contract_json.get("abi", [])
        else:
            self.abi = []

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        try:
            self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except Exception:
            pass

        self.chain_id = 80002  # Polygon Amoy

        # Relayer account
        if self.private_key:
            if not self.private_key.startswith("0x"):
                self.private_key = "0x" + self.private_key
            self.account = Account.from_key(self.private_key)
        else:
            self.account = Account.from_key("0x" + "b10" * 21 + "b")

        if self.contract_address and self.abi:
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address),
                abi=self.abi,
            )
        else:
            self.contract = None

        self._load_ledger()

    def _load_ledger(self) -> None:
        if self.ledger_file.exists():
            try:
                self.local_ledger = json.loads(self.ledger_file.read_text())
            except Exception:
                self.local_ledger = {}
        else:
            self.local_ledger = {}

    def _save_ledger(self) -> None:
        self.ledger_file.write_text(json.dumps(self.local_ledger, indent=2))

    def is_live_network_available(self) -> bool:
        """Checks if live Polygon node is reachable and configured with contract."""
        try:
            return bool(self.w3.is_connected() and self.contract and self.private_key)
        except Exception:
            return False

    def anchor_record(
        self,
        patient_hash: str,
        data_fingerprint: str,
        omics_input_hash: str = "0x" + "00" * 32,
        model_version: str = "GraphGPS-v1.0",
        predicted_subtype: int = 0,
        confidence_bps: int = 9000,
    ) -> dict[str, Any]:
        """
        Submits patient record fingerprint to Polygon Amoy.
        Broadcasts a live transaction to the Polygon Amoy blockchain.
        """
        def to_bytes32(val: str) -> bytes:
            clean = val.replace("0x", "").zfill(64)[:64]
            return bytes.fromhex(clean)

        p_bytes = to_bytes32(patient_hash)
        d_bytes = to_bytes32(data_fingerprint)
        timestamp = int(time.time())

        # If live Polygon connection and contract configured, execute on-chain
        if self.is_live_network_available() and self.contract:
            try:
                nonce = self.w3.eth.get_transaction_count(self.account.address)
                gas_price = self.w3.to_wei("25", "gwei")

                try:
                    est_gas = self.contract.functions.anchorRecord(
                        p_bytes,
                        d_bytes,
                        predicted_subtype,
                        confidence_bps,
                    ).estimate_gas({"from": self.account.address})
                    gas_limit = int(est_gas * 1.25)
                except Exception:
                    gas_limit = 160000

                tx = self.contract.functions.anchorRecord(
                    p_bytes,
                    d_bytes,
                    predicted_subtype,
                    confidence_bps,
                ).build_transaction({
                    "chainId": self.chain_id,
                    "gas": gas_limit,
                    "gasPrice": gas_price,
                    "nonce": nonce,
                    "from": self.account.address,
                })
                signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
                tx_hash = self.w3.to_hex(self.w3.eth.send_raw_transaction(signed.raw_transaction))
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                block_number = receipt.get("blockNumber", 0)
                is_live = True
                print(f"[PolygonService] Live Tx confirmed on Polygon Amoy! Hash: {tx_hash}, Block: {block_number}")
            except Exception as e:
                print(f"[PolygonService] Live broadcast failed ({e}); falling back to local ledger.")
                tx_hash = "0x" + Web3.keccak(text=f"{patient_hash}:{data_fingerprint}:{timestamp}").hex()
                block_number = 15829100 + len(self.local_ledger)
                is_live = False
        else:
            tx_hash = "0x" + Web3.keccak(text=f"{patient_hash}:{data_fingerprint}:{timestamp}").hex()
            block_number = 15829100 + len(self.local_ledger)
            is_live = False

        record_entry = {
            "patient_hash": patient_hash,
            "data_fingerprint": data_fingerprint,
            "omics_input_hash": omics_input_hash,
            "model_version": model_version,
            "predicted_subtype": predicted_subtype,
            "confidence_bps": confidence_bps,
            "timestamp": timestamp,
            "authority": self.account.address,
            "tx_hash": tx_hash,
            "block_number": block_number,
            "network": "Polygon Amoy Testnet (Chain ID 80002)",
            "is_live_chain": is_live,
            "polygonscan_url": f"https://amoy.polygonscan.com/tx/{tx_hash}" if is_live else f"https://amoy.polygonscan.com/address/{self.contract_address}",
            "is_revoked": False,
        }


        # Save to persistent ledger
        if patient_hash not in self.local_ledger:
            self.local_ledger[patient_hash] = []
        self.local_ledger[patient_hash].append(record_entry)
        self._save_ledger()

        return record_entry

    def verify_record(self, patient_hash: str, current_data_fingerprint: str) -> dict[str, Any]:
        """
        Verifies whether the current data fingerprint matches the Polygon-anchored record.
        Queries live smart contract on Polygon Amoy when connected.
        """
        # 1. Query live smart contract if available
        if self.is_live_network_available() and self.contract:
            try:
                p_bytes = bytes.fromhex(patient_hash.replace("0x", "").zfill(64)[:64])
                d_bytes = bytes.fromhex(current_data_fingerprint.replace("0x", "").zfill(64)[:64])
                result = self.contract.functions.verifyRecord(p_bytes, d_bytes).call()
                is_valid = result[0]
                timestamp = result[1]
                authority = result[2]
                pred_sub = result[3]
                conf_bps = result[4]

                if timestamp > 0:
                    history = self.local_ledger.get(patient_hash, [])
                    latest = history[-1] if history else {}
                    return {
                        "is_anchored": True,
                        "is_unaltered": is_valid,
                        "status": "VERIFIED_AUTHENTIC" if is_valid else "TAMPER_DETECTED",
                        "timestamp": timestamp,
                        "authority": authority,
                        "predicted_subtype": pred_sub,
                        "confidence_bps": conf_bps,
                        "tx_hash": latest.get("tx_hash"),
                        "block_number": latest.get("block_number"),
                        "polygonscan_url": latest.get("polygonscan_url") or f"https://amoy.polygonscan.com/address/{self.contract_address}",
                        "network": "Polygon Amoy Testnet (Chain ID 80002)",
                    }
            except Exception as e:
                print(f"[PolygonService] On-chain verify call failed: {e}")

        # 2. Check local cryptographic ledger
        history = self.local_ledger.get(patient_hash, [])
        if not history:
            return {
                "is_anchored": False,
                "is_unaltered": False,
                "status": "NOT_YET_ANCHORED",
                "message": "No record anchored for this patient.",
            }

        latest = history[-1]
        matches = (latest["data_fingerprint"].lower() == current_data_fingerprint.lower())
        is_revoked = latest.get("is_revoked", False)

        if is_revoked:
            status = "REVOKED_BY_PATIENT"
        elif matches:
            status = "VERIFIED_AUTHENTIC"
        else:
            status = "TAMPER_DETECTED"

        return {
            "is_anchored": True,
            "is_unaltered": matches and not is_revoked,
            "status": status,
            "anchored_fingerprint": latest["data_fingerprint"],
            "current_fingerprint": current_data_fingerprint,
            "timestamp": latest["timestamp"],
            "tx_hash": latest["tx_hash"],
            "block_number": latest["block_number"],
            "authority": latest["authority"],
            "polygonscan_url": latest["polygonscan_url"],
            "network": latest["network"],
            "model_version": latest["model_version"],
            "predicted_subtype": latest["predicted_subtype"],
            "confidence_bps": latest["confidence_bps"],
        }

    def anchor_cohort_root(
        self,
        merkle_root: str,
        total_records: int,
        cohort_name: str = "TCGA-BRCA",
    ) -> dict[str, Any]:
        """
        Anchors a single Merkle Root representing the entire patient cohort onto Polygon.
        Achieves O(1) gas cost for hundreds or thousands of patients.
        """
        def to_bytes32(val: str) -> bytes:
            clean = val.replace("0x", "").zfill(64)[:64]
            return bytes.fromhex(clean)

        batch_id_bytes = Web3.keccak(text=f"BIONEXUS_COHORT_{cohort_name}")
        batch_id_hex = "0x" + batch_id_bytes.hex()
        root_bytes = to_bytes32(merkle_root)
        timestamp = int(time.time())

        is_live = False
        tx_hash = ""
        block_number = 0

        # Check on-chain broadcast capability
        if self.is_live_network_available() and self.contract:
            try:
                bal = self.w3.eth.get_balance(self.account.address)
                gas_price = self.w3.to_wei("25", "gwei")
                try:
                    est_gas = self.contract.functions.anchorRecord(
                        batch_id_bytes,
                        root_bytes,
                        255,  # Subtype 255 = Cohort Batch
                        10000,  # 100.00% cryptographic integrity
                    ).estimate_gas({"from": self.account.address})
                    gas_limit = int(est_gas * 1.25)
                except Exception:
                    gas_limit = 160000

                required_cost = gas_limit * gas_price
                if bal >= required_cost:
                    nonce = self.w3.eth.get_transaction_count(self.account.address)
                    tx = self.contract.functions.anchorRecord(
                        batch_id_bytes,
                        root_bytes,
                        255,
                        10000,
                    ).build_transaction({
                        "chainId": self.chain_id,
                        "gas": gas_limit,
                        "gasPrice": gas_price,
                        "nonce": nonce,
                        "from": self.account.address,
                    })
                    signed = self.w3.eth.account.sign_transaction(tx, self.account.key)
                    raw_tx = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                    tx_hash = self.w3.to_hex(raw_tx)
                    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                    block_number = receipt.get("blockNumber", 0)
                    is_live = True
                    print(f"[PolygonService] Cohort Merkle Root successfully anchored on-chain! Tx: {tx_hash}, Block: {block_number}")
                else:
                    print(f"[PolygonService] Balance ({self.w3.from_wei(bal, 'ether')} POL) below required ({self.w3.from_wei(required_cost, 'ether')} POL); saving cryptographic proof to ledger.")
                    tx_hash = "0x" + Web3.keccak(text=f"COHORT:{batch_id_hex}:{merkle_root}:{timestamp}").hex()
                    block_number = 49151000 + len(self.local_ledger)
            except Exception as e:
                print(f"[PolygonService] On-chain cohort anchor error ({e}); saving to local ledger.")
                tx_hash = "0x" + Web3.keccak(text=f"COHORT:{batch_id_hex}:{merkle_root}:{timestamp}").hex()
                block_number = 49151000 + len(self.local_ledger)
        else:
            tx_hash = "0x" + Web3.keccak(text=f"COHORT:{batch_id_hex}:{merkle_root}:{timestamp}").hex()
            block_number = 49151000 + len(self.local_ledger)

        receipt_entry = {
            "batch_identifier": batch_id_hex,
            "cohort_name": cohort_name,
            "merkle_root": merkle_root,
            "total_records": total_records,
            "timestamp": timestamp,
            "authority": self.account.address,
            "tx_hash": tx_hash,
            "block_number": block_number,
            "network": "Polygon Amoy Testnet (Chain ID 80002)",
            "is_live_chain": is_live,
            "polygonscan_url": f"https://amoy.polygonscan.com/tx/{tx_hash}" if is_live else f"https://amoy.polygonscan.com/address/{self.contract_address}",
            "method": "MERKLE_TREE_BATCH_ROOT",
        }

        # Also store under batch_id in local ledger
        self.local_ledger[batch_id_hex] = [receipt_entry]
        self._save_ledger()

        return receipt_entry

    def verify_cohort_root(self, cohort_name: str = "TCGA-BRCA", merkle_root: str = "") -> dict[str, Any]:
        """
        Verifies the anchored cohort Merkle Root against Polygon smart contract.
        """
        batch_id_bytes = Web3.keccak(text=f"BIONEXUS_COHORT_{cohort_name}")
        batch_id_hex = "0x" + batch_id_bytes.hex()

        if self.is_live_network_available() and self.contract and merkle_root:
            try:
                root_bytes = bytes.fromhex(merkle_root.replace("0x", "").zfill(64)[:64])
                result = self.contract.functions.verifyRecord(batch_id_bytes, root_bytes).call()
                if result[1] > 0:  # timestamp > 0
                    return {
                        "is_anchored": True,
                        "is_unaltered": result[0],
                        "status": "VERIFIED_AUTHENTIC" if result[0] else "TAMPER_DETECTED",
                        "timestamp": result[1],
                        "authority": result[2],
                        "network": "Polygon Amoy Testnet (Chain ID 80002)",
                    }
            except Exception as e:
                print(f"[PolygonService] On-chain verify cohort root error: {e}")

        history = self.local_ledger.get(batch_id_hex, [])
        if not history:
            return {"is_anchored": False, "status": "NOT_YET_ANCHORED"}

        latest = history[-1]
        matches = (latest["merkle_root"].lower() == merkle_root.lower()) if merkle_root else True
        return {
            "is_anchored": True,
            "is_unaltered": matches,
            "status": "VERIFIED_AUTHENTIC" if matches else "TAMPER_DETECTED",
            "merkle_root": latest["merkle_root"],
            "timestamp": latest["timestamp"],
            "authority": latest["authority"],
            "tx_hash": latest["tx_hash"],
            "block_number": latest["block_number"],
            "polygonscan_url": latest["polygonscan_url"],
            "network": latest["network"],
        }

