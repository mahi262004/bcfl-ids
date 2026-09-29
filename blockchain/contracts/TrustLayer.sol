// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title TrustLayer
/// @notice Records and scores federated learning client updates for
///         the blockchain-based trust layer in the BCFL-IDS project.
///         Each round, clients submit a hash of their local model
///         update plus a quality/anomaly score computed off-chain.
///         Updates below the reputation threshold are excluded from
///         aggregation on the Python side.
contract TrustLayer {
    address public owner;

    struct ClientRecord {
        uint256 reputation;      // running reputation score (0-100)
        uint256 totalUpdates;
        uint256 acceptedUpdates;
    }

    struct UpdateRecord {
        address client;
        uint256 round;
        bytes32 updateHash;
        uint256 qualityScore;    // 0-100, computed off-chain
        bool accepted;
        uint256 timestamp;
    }

    mapping(address => ClientRecord) public clients;
    UpdateRecord[] public updateLog;

    uint256 public constant REPUTATION_THRESHOLD = 50;
    uint256 public constant INITIAL_REPUTATION = 70;

    event UpdateSubmitted(
        address indexed client,
        uint256 indexed round,
        bytes32 updateHash,
        uint256 qualityScore,
        bool accepted
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized: owner only");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    /// @notice Register a new client (device) with a starting reputation score.
    function registerClient(address clientAddress) external onlyOwner {
        clients[clientAddress].reputation = INITIAL_REPUTATION;
    }

    /// @notice Submit a local model update's hash and off-chain-computed
    ///         quality score. Returns whether it was accepted.
    function submitUpdate(
        uint256 round,
        bytes32 updateHash,
        uint256 qualityScore
    ) external returns (bool accepted) {
        ClientRecord storage record = clients[msg.sender];
        require(record.reputation > 0, "Client not registered");

        accepted = record.reputation >= REPUTATION_THRESHOLD && qualityScore >= REPUTATION_THRESHOLD;

        record.totalUpdates += 1;
        if (accepted) {
            record.acceptedUpdates += 1;
            record.reputation = _clamp(record.reputation + 2);
        } else {
            record.reputation = _clampSub(record.reputation, 10);
        }

        updateLog.push(UpdateRecord({
            client: msg.sender,
            round: round,
            updateHash: updateHash,
            qualityScore: qualityScore,
            accepted: accepted,
            timestamp: block.timestamp
        }));

        emit UpdateSubmitted(msg.sender, round, updateHash, qualityScore, accepted);
    }

    function getReputation(address clientAddress) external view returns (uint256) {
        return clients[clientAddress].reputation;
    }

    function getUpdateLogLength() external view returns (uint256) {
        return updateLog.length;
    }

    function _clamp(uint256 value) internal pure returns (uint256) {
        return value > 100 ? 100 : value;
    }

    function _clampSub(uint256 value, uint256 amount) internal pure returns (uint256) {
        return value > amount ? value - amount : 0;
    }
}
