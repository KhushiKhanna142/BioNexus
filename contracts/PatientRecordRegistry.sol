// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title PatientRecordRegistry
 * @dev Optimized for Polygon Amoy. Anchors cryptographic fingerprints and diagnostic proofs
 * for patient records on Polygon. Raw genomic sequences and PHI are strictly kept off-chain.
 */
contract PatientRecordRegistry {
    struct Record {
        bytes32 dataHash;          // SHA-256 fingerprint of encrypted patient record
        uint256 timestamp;         // Block timestamp when anchored
        address authority;         // Address of clinician/institution
        uint8 predictedSubtype;    // 0: LumA, 1: LumB, 2: Her2, 3: Basal, 4: Normal
        uint16 confidenceBps;      // Confidence in basis points (e.g. 9450 = 94.50%)
    }

    mapping(bytes32 => Record) public latestRecord;
    uint256 public totalAnchored;

    event RecordAnchored(
        bytes32 indexed patientHash,
        bytes32 indexed dataHash,
        uint8 predictedSubtype,
        uint16 confidenceBps,
        address indexed authority,
        uint256 timestamp
    );

    function anchorRecord(
        bytes32 patientHash,
        bytes32 dataHash,
        uint8 predictedSubtype,
        uint16 confidenceBps
    ) external returns (bool) {
        require(patientHash != bytes32(0), "Invalid patient hash");
        require(dataHash != bytes32(0), "Invalid data hash");

        latestRecord[patientHash] = Record({
            dataHash: dataHash,
            timestamp: block.timestamp,
            authority: msg.sender,
            predictedSubtype: predictedSubtype,
            confidenceBps: confidenceBps
        });
        totalAnchored += 1;

        emit RecordAnchored(
            patientHash,
            dataHash,
            predictedSubtype,
            confidenceBps,
            msg.sender,
            block.timestamp
        );
        return true;
    }

    function verifyRecord(
        bytes32 patientHash,
        bytes32 dataHash
    ) external view returns (
        bool isValid,
        uint256 timestamp,
        address authority,
        uint8 subtypeId,
        uint16 confBps
    ) {
        Record memory rec = latestRecord[patientHash];
        if (rec.timestamp > 0 && rec.dataHash == dataHash) {
            return (true, rec.timestamp, rec.authority, rec.predictedSubtype, rec.confidenceBps);
        }
        return (false, rec.timestamp, rec.authority, rec.predictedSubtype, rec.confidenceBps);
    }
}
