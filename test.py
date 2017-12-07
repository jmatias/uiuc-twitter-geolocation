import data.twitter_user as twuser

twitter_users = twuser.load_twitter_users(None, dataset='train')
twitter_users.sort(key=lambda x: x.username)

twitter_users2 = twuser.load_twitter_users(None, dataset='train').sort(key=lambda x: x.username)[0:10000]

print("HEllo")
