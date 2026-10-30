"""Bridge from Python to the TrustLayer contract on a local chain, plus independent audit of records."""
from __future__ import annotations

import json

from .config import BLOCKCHAIN_DIR
from .ledger import canonical_hash

ARTIFACT = BLOCKCHAIN_DIR / "artifacts" / "contracts" / "TrustLayer.sol" / "TrustLayer.json"
DEFAULT_RPC = "http://127.0.0.1:8545"

# The fields that are hashed into record_hash. Reputation is excluded: the chain computes it.
RECORD_FIELDS = ("round", "client", "cosine", "norm_ratio", "score", "gate_passed", "on_probation", "included", "reason", "update_hash")


def chain_available(rpc_url: str = DEFAULT_RPC) -> bool:
    from web3 import HTTPProvider, Web3

    try:
        return Web3(HTTPProvider(rpc_url, request_kwargs={"timeout": 2})).is_connected()
    except Exception:
        return False


class ChainLedger:
    """Same interface as InMemoryLedger, backed by a freshly deployed contract."""

    backend = "chain"

    def __init__(self, rpc_url: str = DEFAULT_RPC) -> None:
        from web3 import HTTPProvider, Web3

        self.w3 = Web3(HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
        if not self.w3.is_connected():
            raise ConnectionError(f"No chain at {rpc_url}")
        self.account = self.w3.eth.accounts[0]
        art = json.loads(ARTIFACT.read_text())
        factory = self.w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        receipt = self._wait(factory.constructor().transact({"from": self.account}))
        self.address = receipt.contractAddress
        self.contract = self.w3.eth.contract(address=self.address, abi=art["abi"])
        self.gas_used = receipt.gasUsed
        self.tx_count = 1

    def _wait(self, tx_hash):
        return self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

    def _send(self, call):
        receipt = self._wait(call.transact({"from": self.account}))
        if receipt.status != 1:
            raise RuntimeError("transaction reverted")
        self.gas_used += receipt.gasUsed
        self.tx_count += 1
        return receipt

    def register(self, client_id: int) -> None:
        self._send(self.contract.functions.registerClient(client_id))

    def submit(self, round_no: int, client_id: int, score: int, update_hash: str, record_hash: str) -> dict:
        receipt = self._send(self.contract.functions.submitScore(round_no, client_id, score, bytes.fromhex(update_hash[2:]), bytes.fromhex(record_hash[2:])))
        count = self.contract.functions.entryCount().call()
        entry = self._entry(count - 1)
        entry["tx_hash"] = receipt.transactionHash.hex()
        entry["block"] = receipt.blockNumber
        return entry

    def anchor_model(self, round_no: int, model_hash: str) -> None:
        self._send(self.contract.functions.anchorModel(round_no, bytes.fromhex(model_hash[2:])))

    def reputation(self, client_id: int) -> int:
        return self.contract.functions.reputation(client_id).call()

    def on_probation(self, client_id: int) -> bool:
        return self.contract.functions.onProbation(client_id).call()

    def _entry(self, index: int) -> dict:
        e = self.contract.functions.entryAt(index).call()
        return {
            "round": e[0], "client": e[1], "score": e[2], "accepted": e[3], "rep_before": e[4], "rep_after": e[5],
            "update_hash": "0x" + e[6].hex().removeprefix("0x"), "record_hash": "0x" + e[7].hex().removeprefix("0x"),
        }

    @property
    def entries(self) -> list[dict]:
        return [self._entry(i) for i in range(self.contract.functions.entryCount().call())]

    @property
    def models(self) -> list[dict]:
        rows = []
        for r in range(1, self.contract.functions.latestRound().call() + 1):
            rows.append({"round": r, "model_hash": "0x" + self.contract.functions.modelHash(r).call().hex().removeprefix("0x")})
        return rows

    def info(self) -> dict:
        return {
            "address": self.address, "oracle": self.account, "transactions": self.tx_count, "gas_used": self.gas_used,
            "block": self.w3.eth.block_number, "chain_id": self.w3.eth.chain_id,
        }


def audit(records: list[dict], entries: list[dict]) -> dict:
    """Recompute each off-chain record's hash and compare it with the ledger. Changes cannot hide."""
    on_ledger = {(e["round"], e["client"]): e for e in entries}
    mismatches = []
    for rec in records:
        key = (rec["round"], rec["client"])
        entry = on_ledger.get(key)
        computed = canonical_hash({f: rec[f] for f in RECORD_FIELDS})
        if entry is None:
            mismatches.append({"round": key[0], "client": key[1], "problem": "no entry on the ledger"})
        elif computed != entry["record_hash"]:
            mismatches.append({"round": key[0], "client": key[1], "problem": "record does not match its on-chain hash", "computed": computed, "on_chain": entry["record_hash"]})
        elif rec["score"] != entry["score"]:
            mismatches.append({"round": key[0], "client": key[1], "problem": "score differs from the chain"})
    return {"checked": len(records), "valid": not mismatches and len(records) == len(entries), "mismatches": mismatches}
