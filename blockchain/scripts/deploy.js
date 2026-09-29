const hre = require("hardhat");

async function main() {
  const TrustLayer = await hre.ethers.getContractFactory("TrustLayer");
  const trustLayer = await TrustLayer.deploy();
  await trustLayer.waitForDeployment();

  console.log("TrustLayer deployed to:", await trustLayer.getAddress());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
