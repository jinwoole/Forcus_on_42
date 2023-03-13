import requests
import json
import config.api_key as api

# Access Token 발급 요청
auth = requests.auth.HTTPBasicAuth(api.client["id"], api.client["secret"])
data = {
    'grant_type': 'client_credentials',
    'client_id': api.client["id"],
    'client_secret': api.client["secret"]
}
headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
}
response = requests.post('https://api.intra.42.fr/oauth/token', auth=auth, data=data, headers=headers)

# 발급된 Access Token
access_token = response.json()['access_token']

# 유저 이름
username = 'seonhoki'

# API 요청 보내기
headers = {
    'Authorization': 'Bearer {}'.format(access_token)
}

campus_id = 29 # Seoul
cursus_id = 21 # ???

response = requests.get('https://api.intra.42.fr/v2/users/{}/titles'.format(username), headers=headers)
response = requests.get(f'https://api.intra.42.fr/v2/cursus_users?filter[campus_id]={campus_id}&filter[cursus_id]={cursus_id}&page[size]=1&page[number]=1', headers=headers)

# API 응답
data = response.json()

print(data)
