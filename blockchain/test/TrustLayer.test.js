const { expect } = require("chai");
const { ethers } = require("hardhat");

const H1 = ethers.id("update");
const H2 = ethers.id("record");

describe("TrustLayer", function () {
  let contract, oracle, client;

  beforeEach(async function () {
    [oracle, client] = await ethers.getSigners();
    contract = await (await ethers.getContractFactory("TrustLayer")).deploy();
    await contract.registerClient(1);
  });

  it("starts a registered client at 70", async function () {
    expect(await contract.reputation(1)).to.equal(70);
    expect(await contract.clientCount()).to.equal(1);
  });

  it("refuses a second registration, so a record cannot be wiped", async function () {
    await expect(contract.registerClient(1)).to.be.revertedWithCustomError(contract, "AlreadyRegistered");
  });

  it("only lets the oracle register clients", async function () {
    await expect(contract.connect(client).registerClient(2)).to.be.revertedWithCustomError(contract, "NotOracle");
  });

  it("stops a client from submitting its own score", async function () {
    await expect(contract.connect(client).submitScore(1, 1, 100, H1, H2)).to.be.revertedWithCustomError(contract, "NotOracle");
  });

  it("stops a client from anchoring a model", async function () {
    await expect(contract.connect(client).anchorModel(1, H1)).to.be.revertedWithCustomError(contract, "NotOracle");
  });

  it("refuses scores above 100", async function () {
    await expect(contract.submitScore(1, 1, 101, H1, H2)).to.be.revertedWithCustomError(contract, "ScoreOutOfRange");
  });

  it("refuses scores for unregistered clients", async function () {
    await expect(contract.submitScore(1, 9, 80, H1, H2)).to.be.revertedWithCustomError(contract, "NotRegistered");
  });

  it("adds 2 on an accepted score and treats 50 as the pass mark", async function () {
    await contract.submitScore(1, 1, 50, H1, H2);
    expect(await contract.reputation(1)).to.equal(72);
    expect((await contract.entryAt(0)).accepted).to.equal(true);
  });

  it("removes 10 on a rejected score and treats 49 as a fail", async function () {
    await contract.submitScore(1, 1, 49, H1, H2);
    expect(await contract.reputation(1)).to.equal(60);
    expect((await contract.entryAt(0)).accepted).to.equal(false);
  });

  it("never goes above 100", async function () {
    for (let i = 0; i < 20; i++) await contract.submitScore(i, 1, 100, H1, H2);
    expect(await contract.reputation(1)).to.equal(100);
  });

  it("never reaches 0, which would mean not registered", async function () {
    for (let i = 0; i < 12; i++) await contract.submitScore(i, 1, 0, H1, H2);
    expect(await contract.reputation(1)).to.equal(1);
    await expect(contract.submitScore(13, 1, 80, H1, H2)).to.not.be.reverted;
  });

  it("puts a client on probation below 50, then lets it recover", async function () {
    for (let i = 0; i < 3; i++) await contract.submitScore(i, 1, 0, H1, H2);
    expect(await contract.reputation(1)).to.equal(40);
    expect(await contract.onProbation(1)).to.equal(true);
    for (let i = 3; i < 8; i++) await contract.submitScore(i, 1, 90, H1, H2);
    expect(await contract.reputation(1)).to.equal(50);
    expect(await contract.onProbation(1)).to.equal(false);
  });

  it("stores the full entry with its hashes", async function () {
    await contract.submitScore(7, 1, 88, H1, H2);
    const e = await contract.entryAt(0);
    expect(e.round).to.equal(7);
    expect(e.client).to.equal(1);
    expect(e.score).to.equal(88);
    expect(e.repBefore).to.equal(70);
    expect(e.repAfter).to.equal(72);
    expect(e.updateHash).to.equal(H1);
    expect(e.recordHash).to.equal(H2);
  });

  it("emits ScoreRecorded", async function () {
    await expect(contract.submitScore(1, 1, 90, H1, H2)).to.emit(contract, "ScoreRecorded").withArgs(1, 1, 90, true, 72, H1, H2);
  });

  it("anchors a model hash once only", async function () {
    await contract.anchorModel(3, H1);
    expect(await contract.modelHash(3)).to.equal(H1);
    expect(await contract.latestRound()).to.equal(3);
    await expect(contract.anchorModel(3, H2)).to.be.revertedWithCustomError(contract, "RoundAlreadyAnchored");
  });

  it("exposes the same constants the Python ledger uses", async function () {
    expect(await contract.REP_START()).to.equal(70);
    expect(await contract.REP_GAIN()).to.equal(2);
    expect(await contract.REP_LOSS()).to.equal(10);
    expect(await contract.REP_FLOOR()).to.equal(1);
    expect(await contract.REP_PROBATION()).to.equal(50);
    expect(await contract.SCORE_PASS()).to.equal(50);
  });
});
