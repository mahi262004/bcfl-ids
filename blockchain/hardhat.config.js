require("@nomicfoundation/hardhat-toolbox");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.19",
  networks: {
    ganache: {
      url: "http://127.0.0.1:8545",
      // Ganache prints 10 pre-funded accounts + private keys on startup.
      // Paste one private key below to deploy from that account, e.g.:
      // accounts: ["0xYOUR_GANACHE_PRIVATE_KEY"],
    },
  },
};
