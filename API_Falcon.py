from waitress import serve
import falcon
from falcon_cors import CORS
import psycopg2
import os
import json
from pandas.io.json import json_normalize
import numpy as np 
import string
from nltk.corpus import stopwords
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline


cors = CORS(allow_all_origins=True, allow_all_headers=True, allow_credentials_all_origins=True, allow_all_methods=True)

conn = psycopg2.connect(
            "dbname='" + "jazz" + "' user='" + 'postgres' + "' host='" + 'localhost' + "' password='" + 'zain2630' + "' port=" + '5433')



df = pd.read_csv('dialogs.txt',sep='\t')
a = pd.Series(df.columns)
a = a.rename({0: df.columns[0],1: df.columns[1]})
b = {'Questions':'Hi','Answers':'hello'}
c = {'Questions':'Hello','Answers':'hi'}
d= {'Questions':'how are you','Answers':"i'm fine. how about yourself?"}
e= {'Questions':'how are you doing','Answers':"i'm fine. how about yourself?"}
f= {'Questions':'what is your name','Answers':"I am rebot!"}
df = df.append(a,ignore_index=True)
df.columns=['Questions','Answers']
df = df.append([b,c,d,e,f],ignore_index=True)
df = df.append(c,ignore_index=True)
df = df.append(d,ignore_index=True)
df = df.append(d,ignore_index=True)

def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]

Pipe = Pipeline([
    ('bow',CountVectorizer(analyzer=cleaner)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])


Pipe.fit(df['Questions'],df['Answers'])



class login:
    _required_params = ['username', 'pass']
    _dot_string = '-----'
    dir = os.path.dirname(__file__)


    def _handleQuery(self, provided_params):
        _required_params = self._required_params
        # Checking whether we are getting all the required parameters. Incomplete parameters will result in an error
        all_params_provided = all([False if param not in provided_params else True for param in _required_params])
        # If we are not getting all the parameters, we gracefully exit with an error statement
        if not all_params_provided:
            return {'Error': 'Missing Parameter. Make Sure all parameters are present. Valid parameters are '
                             '{0}'.format(', '.join(_required_params))}
        username = provided_params['username'] if provided_params['username'] else None
        password = provided_params['pass'] if provided_params['pass'] else None
        print(username, password)
        try:
            query = "Select * from login where username = '"+username +"' and password = '"+password+"'"
            cursor = conn.cursor()
            cursor.execute(query)

            result = cursor.fetchall()
            print(result)
            cursor.close()
            user = result[0][0]
            pa = result[0][1]
            print("here", user, pa)
            return 'login'
        except :    
            return 'Worng Info'
        

    def on_get(self, req, resp):
        params = req.params
        resp.media = self._handleQuery(params)

    def on_post(self, req, resp):
        params = req.media
        resp.media = self._handleQuery(params)


class tabledata:

    _required_params = []
    _dot_string = '-----'
    dir = os.path.dirname(__file__)

    def _handleQuery(self, provided_params):
        print("1234",provided_params)
        _required_params = self._required_params
        # Checking whether we are getting all the required parameters. Incomplete parameters will result in an error
        all_params_provided = all([False if param not in provided_params else True for param in _required_params])
        # If we are not getting all the parameters, we gracefully exit with an error statement
        if not all_params_provided:
            return {'Error': 'Missing Parameter. Make Sure all parameters are present. Valid parameters are '
                             '{0}'.format(', '.join(_required_params))}
      
       
       
        
        
        # resp = main_database.DbResultsQuery("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        # print ("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        query = ("""WITH foobar AS ( 
            SELECT product_id, product_name, description, actual_price, negotiated_price, recommendation_to_admin,id 
            FROM info
            )
            SELECT 
                 json_build_object('data',array_agg(json_build_object('product_id', product_id, 'product_name', product_name,
                                                                      'description', description, 'actual_price', actual_price, 'negotiated_price', negotiated_price, 'recommendation_to_admin', recommendation_to_admin, 'id', id)))
            FROM 
                foobar""")
            
        cursor = conn.cursor()
        cursor.execute(query)

        result = cursor.fetchall()
        print(result)
        cursor.close()
        return result


    def on_get(self, req, resp):
        params = req.params
        resp.media = self._handleQuery(params)

    def on_post(self, req, resp):
        params = req.media
        resp.media = self._handleQuery(params)


class searchid:

    _required_params = ['id']
    _dot_string = '-----'
    dir = os.path.dirname(__file__)

    def _handleQuery(self, provided_params):
        print("1234",provided_params)
        _required_params = self._required_params
        # Checking whether we are getting all the required parameters. Incomplete parameters will result in an error
        all_params_provided = all([False if param not in provided_params else True for param in _required_params])
        # If we are not getting all the parameters, we gracefully exit with an error statement
        if not all_params_provided:
            return {'Error': 'Missing Parameter. Make Sure all parameters are present. Valid parameters are '
                             '{0}'.format(', '.join(_required_params))}
      
       
        search = provided_params['id'] if provided_params['id'] else None
        
        
        # resp = main_database.DbResultsQuery("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        # print ("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        query = ("select * from info where id = {0}".format(search))
            
        cursor = conn.cursor()
        cursor.execute(query)

        result = cursor.fetchall()
        print(result)
        cursor.close()
        try:
            return result[0]
        except:
            return 'Not Found'    


    def on_get(self, req, resp):
        params = req.params
        resp.media = self._handleQuery(params)

    def on_post(self, req, resp):
        params = req.media
        resp.media = self._handleQuery(params)


class update:

    _required_params = ['id','name','dec','no','ac','admin','idd']
    _dot_string = '-----'
    dir = os.path.dirname(__file__)

    def _handleQuery(self, provided_params):
        print("1234",provided_params)
        _required_params = self._required_params
        # Checking whether we are getting all the required parameters. Incomplete parameters will result in an error
        all_params_provided = all([False if param not in provided_params else True for param in _required_params])
        # If we are not getting all the parameters, we gracefully exit with an error statement
        if not all_params_provided:
            return {'Error': 'Missing Parameter. Make Sure all parameters are present. Valid parameters are '
                             '{0}'.format(', '.join(_required_params))}
      
       
        search = provided_params['id'] if provided_params['id'] else None
        name = provided_params['name'] if provided_params['name'] else None
        dec = provided_params['dec'] if provided_params['dec'] else None
        no = provided_params['no'] if provided_params['no'] else None
        ac = provided_params['ac'] if provided_params['ac'] else None
        admin = provided_params['admin'] if provided_params['admin'] else None
        idd = provided_params['idd'] if provided_params['idd'] else None
        
        
        # resp = main_database.DbResultsQuery("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        # print ("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        query = ("update info set product_name = '{0}' , description = '{1}', actual_price = {2}, negotiated_price = {3}, recommendation_to_admin =  '{4}' , product_id = {5} where id = {6}".format(name, dec, ac, no, admin, search,idd))           
        print(query)
        cursor = conn.cursor()
        cursor.execute(query)
        cursor.close()
        return 'Successfully updated'


    def on_get(self, req, resp):
        params = req.params
        resp.media = self._handleQuery(params)

    def on_post(self, req, resp):
        params = req.media
        resp.media = self._handleQuery(params)   


class dec:
    _required_params = ['pro']
    _dot_string = '-----'
    dir = os.path.dirname(__file__)


    def _handleQuery(self, provided_params):
        _required_params = self._required_params
        # Checking whether we are getting all the required parameters. Incomplete parameters will result in an error
        all_params_provided = all([False if param not in provided_params else True for param in _required_params])
        # If we are not getting all the parameters, we gracefully exit with an error statement
        if not all_params_provided:
            return {'Error': 'Missing Parameter. Make Sure all parameters are present. Valid parameters are '
                             '{0}'.format(', '.join(_required_params))}
        pro = provided_params['pro'] if provided_params['pro'] else None
       
      
        query = "select decription from dectab where product = '{0}'".format(pro)
        cursor = conn.cursor()
        cursor.execute(query)

        result = cursor.fetchall()
        print(result[0][0])
        result = result[0][0]
        cursor.close()
        return result
    
        

    def on_get(self, req, resp):
        params = req.params
        resp.media = self._handleQuery(params)

    def on_post(self, req, resp):
        params = req.media
        resp.media = self._handleQuery(params)


class chat:
    _required_params = ['tex']
    _dot_string = '-----'
    dir = os.path.dirname(__file__)


    def _handleQuery(self, provided_params):
        _required_params = self._required_params
        # Checking whether we are getting all the required parameters. Incomplete parameters will result in an error
        all_params_provided = all([False if param not in provided_params else True for param in _required_params])
        # If we are not getting all the parameters, we gracefully exit with an error statement
        if not all_params_provided:
            return {'Error': 'Missing Parameter. Make Sure all parameters are present. Valid parameters are '
                             '{0}'.format(', '.join(_required_params))}
        tex = provided_params['tex'] if provided_params['tex'] else None
       
      
        
        ans=Pipe.predict([tex])[0]
        return ans
    
        

    def on_get(self, req, resp):
        params = req.params
        resp.media = self._handleQuery(params)

    def on_post(self, req, resp):
        params = req.media
        resp.media = self._handleQuery(params)  


class searchname:

    _required_params = ['name']
    _dot_string = '-----'
    dir = os.path.dirname(__file__)

    def _handleQuery(self, provided_params):
        print("1234",provided_params)
        _required_params = self._required_params
        # Checking whether we are getting all the required parameters. Incomplete parameters will result in an error
        all_params_provided = all([False if param not in provided_params else True for param in _required_params])
        # If we are not getting all the parameters, we gracefully exit with an error statement
        if not all_params_provided:
            return {'Error': 'Missing Parameter. Make Sure all parameters are present. Valid parameters are '
                             '{0}'.format(', '.join(_required_params))}
      
       
        name = provided_params['name'] if provided_params['name'] else None
        
        
        # resp = main_database.DbResultsQuery("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        # print ("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        query = ("select * from info where product_name = '{0}'".format(name))
            
        cursor = conn.cursor()
        cursor.execute(query)

        result = cursor.fetchall()
        print(result)
        cursor.close()
        try:
            return result[0]
        except:
            return 'Not Found'    


    def on_get(self, req, resp):
        params = req.params
        resp.media = self._handleQuery(params)

    def on_post(self, req, resp):
        params = req.media
        resp.media = self._handleQuery(params)


class botorder:

    _required_params = ['admin','des','ac','no','id','name']
    _dot_string = '-----'
    dir = os.path.dirname(__file__)

    def _handleQuery(self, provided_params):
        print("1234",provided_params)
        _required_params = self._required_params
        # Checking whether we are getting all the required parameters. Incomplete parameters will result in an error
        all_params_provided = all([False if param not in provided_params else True for param in _required_params])
        # If we are not getting all the parameters, we gracefully exit with an error statement
        if not all_params_provided:
            return {'Error': 'Missing Parameter. Make Sure all parameters are present. Valid parameters are '
                             '{0}'.format(', '.join(_required_params))}
      
       
        admin = provided_params['admin'] if provided_params['admin'] else None
        des = provided_params['des'] if provided_params['des'] else None
        ac = provided_params['ac'] if provided_params['ac'] else None
        no = provided_params['no'] if provided_params['no'] else None
        idd = provided_params['id'] if provided_params['id'] else None
        name = provided_params['name'] if provided_params['name'] else None
        
        
        # resp = main_database.DbResultsQuery("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        # print ("""select id, username, plugin, permission_status, granted_at::varchar from plugin_permit_status""")
        query = ("insert into info (recommendation_to_admin, description,actual_price,negotiated_price, product_id,  product_name)  values('{0}', '{1}', '{2}', '{3}', '{4}', '{5}');".format(admin, des, ac, no, idd, name))
        print(query) 
        cursor = conn.cursor()
        print(cursor)
        cursor.execute(query)
        conn.commit()

        return "Success"
            


    def on_get(self, req, resp):
        params = req.params
        resp.media = self._handleQuery(params)

    def on_post(self, req, resp):
        params = req.media
        resp.media = self._handleQuery(params)                                      

        



if __name__ == '__main__':
    api = falcon.API(middleware=[cors.middleware])
    api.add_route('/data', login())
    api.add_route('/tabledata', tabledata())
    api.add_route('/id', searchid())
    api.add_route('/update', update())
    api.add_route('/dec', dec())
    api.add_route('/chat', chat())
    api.add_route('/name', searchname())
    api.add_route('/bot', botorder())

    serve(api, host='0.0.0.0', port=8090)