"""
Bridges the Python FL side to the deployed TrustLayer Solidity contract.

Fill in CONTRACT_ADDRESS after running:
    cd blockchain && npm run deploy:ganache
(the deploy script prints the deployed address; the ABI is generated
at blockchain/artifacts/contracts/TrustLayer.sol/TrustLayer.json)

Run/import from the project root so the ABI path below resolves correctly.
"""

import hashlib
import json
from pathlib import Path

from web3 import Web3

GANACHE_URL = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = None  # e.g. "0xABC123..." after deployment
ABI_PATH = Path("blockchain/artifacts/contracts/TrustLayer.sol/TrustLayer.json")


def get_contract():
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    assert w3.is_connected(), "Could not connect to Ganache -- is it running?"

    with open(ABI_PATH) as f:
        abi = json.load(f)["abi"]

    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
    return w3, contract


def hash_update(update_array) -> bytes:
    """Turns a client's model update (numpy array) into a fixed-size
    hash to log on-chain -- the raw weights themselves never go on
    the blockchain, only a fingerprint of them."""
    return hashlib.sha256(update_array.tobytes()).digest()


def submit_update(w3, contract, account, round_num: int, update_hash: bytes, quality_score: int) -> bool:
    """Submits an update's hash + quality score to the smart contract
    and returns whether it was accepted (reputation/quality check)."""
    tx_hash = contract.functions.submitUpdate(
        round_num, update_hash, quality_score
    ).transact({"from": account})
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    event = contract.events.UpdateSubmitted().process_receipt(receipt)
    return event[0]["args"]["accepted"] if event else False
