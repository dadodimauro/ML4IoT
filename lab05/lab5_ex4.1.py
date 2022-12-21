import cherrypy
import json
import redis
import uuid
from redis.commands.json.path import Path


REDIS_HOST = "redis-18937.c72.eu-west-1-2.ec2.cloud.redislabs.com"
REDIS_PORT = 18937
REDIS_USER = "default"
REDIS_PASSWORD = "DlfmUPWr2iMKAbzvEiwzLCizwt2yjLkP"

# Connect to Redis server
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USER, password=REDIS_PASSWORD)
is_connected = redis_client.ping()
print('Redis Connected:', is_connected)

# endpoint /status
class Status(object):
    exposed = True  # make the endopoint reachable to the user

    def GET(self, *path, **query):
        response_dict = {
            'status': 'online'
        }
        response = json.dumps(response_dict)

        return response

# endpoint /todos
class TodoList(object):
    exposed = True  # make the endopoint reachable to the user

    def GET(self, *path, **query):
        # print(query)
        keys = redis_client.keys('todo:*')
        print(keys)
        items = []

        completed = query.get('completed', None)  # None is the default value (if something is wrong)
        if completed is not None:
            completed = bool(completed)

        message = query.get('message', None)

        for key in keys:
            key = key.decode()  # key = "todo:"70391b99-c9f3-4903-9043-00195d0c970b"
            item = redis_client.json().get(key)
            # item = 
            # {
            #   "message": "Byu Coffe",
            #   "completed" : False
            # }
            uid = key.removeprefix('todo:')  # uid = "70391b99-c9f3-4903-9043-00195d0c970b"
            item['id'] = uid
            # item = 
            # {
            #   "message": "Byu Coffe",
            #   "completed" : False,
            #   "id": "70391b99-c9f3-4903-9043-00195d0c970b"
            # }

            if completed is not None and message is not None:
                if completed == item['completed'] and message in item['message']:
                    items.append(item)  # update the list
            elif completed is not None:
                if completed == item['completed']:
                    items.append(item)  # update the list
            elif message is not None:
                if message in item['message']:
                    items.append(item)  # update the list
            else:
                items.append(item)  # update the list

        response = json.dumps(items)

        return response

    def POST(self, *path, **query):
        uid = uuid.uuid4()
        body = cherrypy.request.body.read()  # get the request body
        # print(body)
        body_dict = json.loads(body.decode())
        # print(body_dict)

        todo_data = {
            'message': body_dict['message'],
            'completed': False,  # the users can't specify the message in the POST method
        }
        redis_client.json().set(f'todo:{uid}', Path.root_path(), todo_data)
        
        return

# retreive a specific TODO item
# endpoint /todo/{id}
class TodoDetail(object):
    exposed = True

    def GET(self, *path, **query):  # path parameter is REQUIRED, the user must specify it
        if len(path) != 1:  # we can only have the message id
            raise cherrypy.HTTPError(400, 'Bad Request: missing id')
        uid = path[0]

        item = redis_client.json().get(f'todo:{path[0]}')  # if the key is invalid item will be equal to None

        if item is None:
             raise cherrypy.HTTPError(404, '404 Not Found')
        
        item['id'] = uid

        response = json.dumps(item)
        
        return response

    def PUT(self, *path, **query):
        if len(path) != 1:
            raise cherrypy.HTTPError(400, 'Bad Request: missing id')
        uid = path[0]

        item = redis_client.json().get(f'todo:{path[0]}')

        if item is None:
             raise cherrypy.HTTPError(404, '404 Not Found')
        
        body = cherrypy.request.body.read()
        body_dict = json.loads(body.decode())

        redis_client.json().set(f'todo:{uid}', Path.root_path(), body_dict)
        
        return

    def DELETE(self, *path, **query):
        if len(path) != 1:
            raise cherrypy.HTTPError(400, 'Bad Request: missing id')
        uid = path[0]
        
        found = redis_client.delete(f'todo:{uid}')

        if found == 0:
            raise cherrypy.HTTPError(404, '404 Not Found')

        return

if __name__ == '__main__':
    # evertything after "/" in the URL must be sent to different functions
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    # cherrypy.tree.mount(Status(), '/status', conf) /status is wrong because in the documentation the name is different 
    cherrypy.tree.mount(Status(), '/online', conf)  # endpoint routing to the online object
    cherrypy.tree.mount(TodoList(), '/todos', conf)
    cherrypy.tree.mount(TodoDetail(), '/todo', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()