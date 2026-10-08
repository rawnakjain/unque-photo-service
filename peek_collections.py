import pymongo
import json

uri = "mongodb+srv://praunak_db_user:mXxGITnLvIwEaRiB@photo-service.1sd1wva.mongodb.net/?appName=photo-service"
client = pymongo.MongoClient(uri)
db = client["Unque"]

print("=== salonservices sample ===")
for doc in db.salonservices.find().limit(3):
    print(json.dumps(doc, indent=2, default=str))

print("\n=== salonservicevariations sample ===")
for doc in db.salonservicevariations.find().limit(3):
    print(json.dumps(doc, indent=2, default=str))

client.close()