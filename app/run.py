from application.dash_code import app
from settings import config
app.run_server(debug=config.debug, host=config.host, port=config.port)

