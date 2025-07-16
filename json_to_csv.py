import json
import csv

def convert_to_csv(transaction):
    DATA = {
        "_id": transaction["_id"].get("$oid"),
        "userWallet": transaction["userWallet"],
        "network": transaction["network"],
        "protocol": transaction["protocol"],
        "txHash": transaction["txHash"],
        "logId": transaction["logId"],
        "timestamp": transaction["timestamp"],
        "blockNumber": transaction["blockNumber"],
        "action": transaction["action"],
        "type": transaction["actionData"].get("type"),
        "amount": transaction["actionData"].get("amount"),
        "assetSymbol": transaction["actionData"].get("assetSymbol"),
        "assetPriceUSD": transaction["actionData"].get("assetPriceUSD"),
        "poolId": transaction["actionData"].get("poolId"),
        "userId": transaction["actionData"].get("userId"),
        "createdAt": transaction["createdAt"]["$date"],
        "updatedAt": transaction["updatedAt"]["$date"]
    }
    return DATA

def converter(json_file:str,csv_file:str):
    with open(json_file,"r") as j:
        data = json.load(j)

    tx_history = [convert_to_csv(tx) for tx in data]
    with open(csv_file, "w") as c:
        writer = csv.DictWriter(c,fieldnames=tx_history[0].keys())
        writer.writeheader()
        writer.writerows(tx_history)
    print(f"converted json file to CSV for Machine Learning, file name is {csv_file}")

converter("user-wallet-transactions.json","user-wallet-transactions.csv")
