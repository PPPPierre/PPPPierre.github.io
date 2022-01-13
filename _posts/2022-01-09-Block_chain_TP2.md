---
title: 区块链 Lab - Smart Contracts
author: Stone SHI
date: 2022-01-09 15:53:00 +0200
categories: [Blogging, Block chain, Smart contracts]
tags: [Block chain, Smart contracts]
---

# 1 Guidelines
SmartPy is a high level language (implemented in a Python library) that lets you write smart contracts for Tezos in Python. Furthermore, the website of the project, https://smartpy.io contains handy tools to write, test, deploy smart contracts and interact with them. In this lab, we will use SmartPy and its online platform.

## 1.1 Do not lose your data!
- SmartPy uses cookies to save locally in your browser your data (scripts, wallets, etc.) Make sure not to delete them! (Your browser may automatically delete them when you close it.)

- You may also want to save your data on your disk: scripts, faucet data (mnemonics in JSON file), account and contract addresses (tz1... and KT1...), etc.

## 1.2 Links
- Slides and Lab exercises https://onurb.gitlab.io/mines2022
- SmartPy
    - Online editor https://smartpy.io/ide
    - Wallet https://smartpy.io/wallet/
    - Contract explorer https://smartpy.io/explorer.html
    - Reference manual https://smartpy.io/docs/ and in particular these parts:
        - the Tests and Scenarios section and the example https://smartpy.io/docs/scenarios/framework#test-example
        - Checking conditions https://smartpy.io/docs/general/checking_condition
- Block explorers:
    - TzKT https://hangzhounet.tzkt.io
    - tzStats https://hangzhou.tzstats.com

## 1.3 E-mail me your work
- Python scripts and `tz1...` and `KT1...` addresses of Hangzhounet accounts used, answers to questions.
- My email is bruno@nomadic-labs.com.

# 2 Getting familiar with SmartPy

# 2.1 First steps
1. Open the SmartPy online editor at https://smartpy.io/ide. Close the pop up, you should have an empty program on your left.

2. Copy-paste the Welcome template available at (https://onurb.gitlab.io/mines2022/welcome.py).

```python
import smartpy as sp

# This is the SmartPy editor.
# You can experiment with SmartPy by loading a template.
# (in the Commands menu above this editor)
#
# A typical SmartPy program has the following form:

# A class of contracts
class MyContract(sp.Contract):
    def __init__(self, myParameter1, myParameter2):
        self.init(myParameter1 = myParameter1,
                  myParameter2 = myParameter2)

    # An entry point, i.e., a message receiver
    # (contracts react to messages)
    @sp.entry_point
    def myEntryPoint(self, params):
        sp.verify(self.data.myParameter1 <= self.data.myParameter2)
        self.data.myParameter1 += params

# Tests
@sp.add_test(name = "Welcome")
def test():
    # We define a test scenario, together with some outputs and checks
    scenario = sp.test_scenario()

    # We first define a contract and add it to the scenario
    c1 = MyContract(12, 123)
    scenario += c1

    # And call some of its entry points
    scenario += c1.myEntryPoint(12)
```
3. Run the contract (White on blue play symbol). Observe the new elements displayed on the right panel, in particular the SmartPy and Types tabs.

4. An entrypoint is a method that can be called when interacting with the contract. What does myEntryPoint do? In particular, what does line 19 test?

line 19: `sp.verify(self.data.myParameter1 <= self.data.myParameter2)`

```python
import smartpy as sp

# This is the SmartPy editor.
# You can experiment with SmartPy by loading a template.
# (in the Commands menu above this editor)
#
# A typical SmartPy program has the following form:

# A class of contracts
class MyContract(sp.Contract):
    def __init__(self, myParameter1, myParameter2):
        self.init(myParameter1 = myParameter1,
                  myParameter2 = myParameter2)

    # An entry point, i.e., a message receiver
    # (contracts react to messages)
    @sp.entry_point
    def myEntryPoint(self, params):
        sp.verify(self.data.myParameter1 <= self.data.myParameter2)
        self.data.myParameter1 += params

    @sp.entry_point
    def reset(self, params):
        self.data.myParameter1 = 0
        self.data.myParameter2 = params

# Tests
@sp.add_test(name = "Welcome")
def test():
    # We define a test scenario, together with some outputs and checks
    scenario = sp.test_scenario()

    # We first define a contract and add it to the scenario
    c1 = MyContract(123, 12)
    scenario += c1

    # And call some of its entry points
    scenario += c1.reset(123)
    scenario += c1.myEntryPoint(12)
```

# 3 MinesCoin

You want to develop a new blockchain, `MinesCoin` that promises to be revolutionary. In order to finance its development, you launch a fundraiser by pre-saling the tokens of your future blockchain. The `MinesCoin` fundraiser will be managed by a Tezos smart contract.

## 3.1 Setting up accounts

Download the template from https://onurb.gitlab.io/mines2022/minescoin.py and copy-paste it in the SmartPy online editor.

```python
import smartpy as sp

class MinesCoin(sp.Contract):
    def __init__(self, admin):
        self.init(balances = sp.big_map(), administrator = admin, totalSupply = 0)

    @sp.entry_point
    def transfer(self, params):
        sp.verify((params.origin == sp.sender) |
                  (self.data.balances[params.origin].approvals[sp.sender] >= params.amount))
        self.addAddressIfNecessary(params.destination)
        sp.verify(self.data.balances[params.origin].balance >= params.amount)
        self.data.balances[params.origin].balance -= params.amount
        self.data.balances[params.destination].balance += params.amount
        sp.if (params.origin != sp.sender):
            self.data.balances[params.origin].approvals[sp.sender] -= params.amount

    @sp.entry_point
    def approve(self, params):
        sp.verify(params.origin == sp.sender)
        alreadyApproved = self.data.balances[params.origin].approvals.get(params.destination, 0)
        sp.verify((alreadyApproved == 0) | (params.amount == 0))
        self.data.balances[params.origin].approvals[params.destination] = params.amount

    @sp.entry_point
    def mint(self, params):
        sp.verify(sp.sender == self.data.administrator)
        self.addAddressIfNecessary(params.address)
        self.data.balances[params.address].balance += params.amount
        self.data.totalSupply += params.amount

    def addAddressIfNecessary(self, address):
        sp.if ~ self.data.balances.contains(address):
            self.data.balances[address] = sp.record(balance = 0, approvals = {})

@sp.add_test(name = "MinesCoin")
def test():

    scenario = sp.test_scenario()
    scenario.h1("Simple FA12 Contract")

    scenario.h2("Accounts")
    # sp.test_account generates ED25519 key-pairs deterministically:
    admin = sp.test_account("Administrator")

    # Let's display the accounts:
    # scenario.show([admin, alice, bob, charlie])

    c1 = MinesCoin(admin.address)

    scenario += c1

    scenario.h2("Minting coins")
    scenario.h3("TODO Admin mints 18 coins for Alice")
    # scenario += c1.mint(address = ..., amount = ...).run(sender = ...)
    scenario.h3("TODO Admin mints 24 coins for Bob")

    scenario.h2("Transfers")
    scenario.h3("TODO Alice transfers directly 4 tokens to Bob")
    scenario.h3("TODO Charlie tries to transfer from Alice but does not have her approval")
    scenario.h3("TODO Alice approves Charlie")
    scenario.h3("TODO Charlie transfers Alice's tokens to Bob")
    scenario.h3("TODO Charlie tries to over transfer from Alice")
    scenario.h3("TODO Alice removes the approval for Charlie")
    scenario.h3("TODO Charlie tries to transfer Alice's tokens to Bob and fails")
    scenario.h2("Burning coins")
    scenario.h3("TODO Admin burns Bob token")
    scenario.h3("TODO Alice tries to burn Bob token")
```

1. Run it to verify that the template is fine.

2. An administrator account is already set. Create three accounts for Alice, Bob and Charlie.

3. Uncomment the line about the display of accounts. Run the contract.

```python
    # ...

    scenario.h2("Accounts")
    # sp.test_account generates ED25519 key-pairs deterministically:
    admin = sp.test_account("Administrator")
    alice = sp.test_account("Alice")
    bob = sp.test_account("Bob")
    charlie = sp.test_account("Charlie")

    # Let's display the accounts:
    scenario.show([admin, alice, bob, charlie])

    c1 = MinesCoin(admin.address)

    # ...
```

## 3.2 Minting coins
Alice and Bob are super excited by `MinesCoin` and they participated in your crowdfunding campaign. As a consequence, you will credit them with some coins.

In order to do that you need to use the `mint` entry point. `mint` has two parameters `address` and `amount`, which represents respectively **the address that is credited** and **the amount of coins created**.

4. Read the code for the `mint` entry point. Which account has the right to mint coins? What happens if the address given as argument does not exist?
5. Uncomment the line calling mint and then fill the dots so that 18 coins are minted for Alice. Which account should be the sender of the call to `mint`? (Tip: check the documentation on Test accounts https://smartpy.io/docs/scenarios/testing#test-accounts, `address` and `sender` are addresses.)
6. Write the call to `mint` that creates 24 coins for Bob.
7. Verify that the total supply is 42. (Tip: use `scenario.verify(..)`.)

```python
    # ...

    scenario.h2("Minting coins")
    scenario.h3("TODO Admin mints 18 coins for Alice")
    scenario += c1.mint(address = alice.address, amount = 18).run(sender = admin.address)
    scenario.h3("TODO Admin mints 24 coins for Bob")
    scenario += c1.mint(address = bob.address, amount = 24).run(sender = admin.address)
    scenario.verify(c1.data.totalSupply == 42)

    # ...
```

# 3.3 Transfers and approvals
You want to allow transfers of `MinesCoin` between contributors while the technology is not ready yet.

8. Read the code of transfer. What are its three arguments?

`params.origin`, `params.amount` and `params.destination`.

Direct and indirect transfers are possible. Direct transfers are simple: Alice calls `transfer` to send some of her tokens to Bob. Indirect transfers are a bit more subtle: Charlie calls `transfer` to send `X` of Alice's coins to Bob. This indirect transfer will succeed if Alice has approved that Charlie can spend `X` or more tokens on her behalf. Indirect transfers are useful if Charlie is a third party (exchange, wallet, etc.) that can connect Alice and Bob.

9. Write the call to transfer that corresponds to Alice sending directly 4 tokens to Bob.
10. Write the call to transfer that corresponds to Charlie sending 8 tokens from Alice to Bob. Does it fail? Why? Fix this by giving Alice's approval to Charlie for spending 10 of her tokens. You should use a call to approve.
11. Write another call where Charlie tries to over-transfer from Alice.
12. Remove the approval of Alice to Charlie by calling approve with an amount of 0.
13. Verify that Charlie cannot indeed transfer from Alice.

# 3.4 Burning coins

Alas, developing a blockchain technology turns out much harder than what you expected. `MinesCoin` is still not ready and some contributors want some of their money back. The rules of the fundraiser allow that: contributions are given back and tokens are destroyed or **burned**.

12. Write a `burn` entry point that subtracts tokens to the balances of accounts. Like `mint`, `burn` has two parameters `address` and `amount`. Note that
    - `burn` must be called by the administrator
    - that the amount of burnt tokens must be smaller or equal to the balance of the account
    - that the `totalSupply` needs to be updated as well.
13. Write a test where Admin burns 2 tokens from Bob. Verify that the balance of Bob has decreased.
14. Write a test that verifies that Alice cannot burn Bob tokens.
15. Write a test that verifies that no token is burnt if the amount is more than the number of tokens possessed by Bob.

## 3.5 Deploying it on Hangzhounet
16. Use the faucet to create accounts for the Administrator, Alice, Bob and Charlie.
17. Deploy the contract with the Administrator signing the operation. Note that the initial storage is defined by the test scenario and cannot be changed. Add in the test scenario a new instance of the contract with the Administrator from the Faucet. (Use `sp.address('tz1...')`). Save the originated contract in your wallet. (Tip: you may need to reload the `estimate cost from RPC` if your gas limit is too low.)
18. Interact with the contract using the Explorer (https://smartpy.io/explorer.html). You can replay the tests previously used and observe the modification of the storage. (Tip: have two windows opened, one with the explorer, another with the wallet so you can quickly find your wallet addresses.)

Check the status of the contract in using https://hangzhou.tzstats.com/, because the Explorer can not correctly show the data `balances`.