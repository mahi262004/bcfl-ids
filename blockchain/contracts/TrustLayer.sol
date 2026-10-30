// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title TrustLayer
/// @notice Tamper-evident record of the federated server's trust decisions.
/// The server is the only oracle, so this contract makes its verdicts auditable after the
/// fact; it does not decentralise the scoring itself.
contract TrustLayer {
    uint8 public constant REP_START = 70;
    uint8 public constant REP_GAIN = 2;
    uint8 public constant REP_LOSS = 10;
    uint8 public constant REP_FLOOR = 1;
    uint8 public constant REP_CAP = 100;
    uint8 public constant REP_PROBATION = 50;
    uint8 public constant SCORE_PASS = 50;

    struct Entry {
        uint32 round;
        uint32 client;
        uint8 score;
        bool accepted;
        uint8 repBefore;
        uint8 repAfter;
        bytes32 updateHash; // hash of the client's model update
        bytes32 recordHash; // hash of the full trust explanation record
    }

    address public immutable oracle;
    mapping(uint32 => uint8) public reputation; // 0 means not registered
    uint32[] public clients;
    Entry[] private entries;
    mapping(uint32 => bytes32) public modelHash; // round => hash of the global model
    uint32 public latestRound;

    event ClientRegistered(uint32 indexed client);
    event ScoreRecorded(uint32 indexed round, uint32 indexed client, uint8 score, bool accepted, uint8 repAfter, bytes32 updateHash, bytes32 recordHash);
    event ModelAnchored(uint32 indexed round, bytes32 modelHash);

    error NotOracle();
    error AlreadyRegistered();
    error NotRegistered();
    error ScoreOutOfRange();
    error RoundAlreadyAnchored();

    modifier onlyOracle() {
        if (msg.sender != oracle) revert NotOracle();
        _;
    }

    constructor() {
        oracle = msg.sender;
    }

    /// @notice A client registers once, so a punished client cannot wipe its record.
    function registerClient(uint32 client) external onlyOracle {
        if (reputation[client] != 0) revert AlreadyRegistered();
        reputation[client] = REP_START;
        clients.push(client);
        emit ClientRegistered(client);
    }

    /// @notice Only the server submits scores, so a client cannot award itself a perfect one.
    function submitScore(uint32 round, uint32 client, uint8 score, bytes32 updateHash, bytes32 recordHash) external onlyOracle {
        uint8 before = reputation[client];
        if (before == 0) revert NotRegistered();
        if (score > 100) revert ScoreOutOfRange();
        bool accepted = score >= SCORE_PASS;
        uint8 next;
        if (accepted) {
            // A client on probation can earn its way back with good updates.
            next = before + REP_GAIN > REP_CAP ? REP_CAP : before + REP_GAIN;
        } else {
            // Reputation stops at 1, because 0 means not registered.
            next = before <= REP_FLOOR + REP_LOSS ? REP_FLOOR : before - REP_LOSS;
        }
        reputation[client] = next;
        entries.push(Entry(round, client, score, accepted, before, next, updateHash, recordHash));
        emit ScoreRecorded(round, client, score, accepted, next, updateHash, recordHash);
    }

    /// @notice Pin the global model of a round. It cannot be replaced afterwards.
    function anchorModel(uint32 round, bytes32 hash) external onlyOracle {
        if (modelHash[round] != bytes32(0)) revert RoundAlreadyAnchored();
        modelHash[round] = hash;
        if (round > latestRound) latestRound = round;
        emit ModelAnchored(round, hash);
    }

    function onProbation(uint32 client) external view returns (bool) {
        uint8 rep = reputation[client];
        return rep != 0 && rep < REP_PROBATION;
    }

    function entryCount() external view returns (uint256) {
        return entries.length;
    }

    function entryAt(uint256 index) external view returns (Entry memory) {
        return entries[index];
    }

    function clientCount() external view returns (uint256) {
        return clients.length;
    }
}
