import requests
import json
import config.api_key as api
import pprint as pp

# 선택된 Title 가져오기, 없다면 공백 return
def get_title(data):
    title = ''
    i = 0
    for t in data['titles_users']:
        if t['selected'] is True:
            title = data['titles'][i]['name']
            return title
        i += 1
    return title

# user_id를 받아 필요한 data를 dict로 return
def get_user_data(user_id):
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

    # API 요청 보내기
    headers = {
        'Authorization': 'Bearer {}'.format(access_token)
    }

    response = requests.get(f'https://api.intra.42.fr/v2/users/{user_id}', headers=headers)

    # API 응답
    data = response.json()
    title = get_title(data)

    res = {
        'user_id': data['login'],
        'kind': data['kind'],
        'is_staff': data['staff?'],
        'is_active': data['active?'],
        'grade': data['cursus_users'][1]['grade'],
        'title': title,
    }
    return res

def main():
    user_id = "juha"
    user_data = get_user_data(user_id)
    pp.pprint(user_data)

    user_id = "jinwoole"
    user_data = get_user_data(user_id)
    pp.pprint(user_data)

    user_id = "seonhoki"
    user_data = get_user_data(user_id)
    pp.pprint(user_data)

    user_id = "jekim"
    user_data = get_user_data(user_id)
    pp.pprint(user_data)

main()