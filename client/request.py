from intra import IntraAPIClient
ic = IntraAPIClient()

response = ic.get("cursus_id/2")
data = response.json()
print(data)


