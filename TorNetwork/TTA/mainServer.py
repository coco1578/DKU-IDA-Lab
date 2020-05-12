import json
import time
import struct
import logging
import threading
import socketserver

import mlServer
import torDB

logger = logging.getLogger('mainServer')
logger.setLevel(logging.DEBUG)
frmt = '[%(asctime)-15s][%(levelname)s][%(module)s][%(funcName)s] %(message)s'
logging.basicConfig(format=frmt)

class TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
	allow_reuse_address = True

	def __init__(self, server_address, RequestHandlerClass):
		super(socketserver.ThreadingMixIn, self).__init__(server_address, RequestHandlerClass)
		t0 = time.time()
		self.mlDiscriminator = mlServer.mlDiscriminator()
		print('mlDiscriminator is loaded', time.time() - t0)

		t0 = time.time()
		self.db = torDB.DBSearch()
		print('tor Database is loaded', time.time() - t0)


class TCPServerHandler(socketserver.BaseRequestHandler):

    def handle(self):
        try:
            # TODO:// time_synk will use!!
            client_ip, time_synk, packet_features = self.read_file()
            cur_thread = threading.current_thread()
            logger.info('Client connected {0}: {1}'.format(cur_thread.name, client_ip))

            # ML predict
            y_pred = self.server.mlDiscriminator.is_tor_or_non_tor(packet_features)

            if y_pred == 0:  # Non tor
                self.send_obj({'tor': 'no'})
                logger.info('This data is non tor data')
            elif y_pred == 1:  # Tor or Hidden service
                y_pred = self.server.mlDiscriminator.is_tor_or_hidden_service(packet_features)

                if y_pred == 0:  # Tor
                    y_pred = self.server.mlDiscriminator.close_world_tor_site(packet_features)
                    table_name = 'tor'

                    result_url, contents = self.server.db.get_url_query(table_name, y_pred)

                    self.send_obj({'tor': 'tor','client_ip': client_ip, 'url': result_url, 'contents': contents})
                    logger.info('This data is tor', result_url)

                else:  # Hidden
                    y_pred = self.server.mlDiscriminator.close_world_hidden_site(packet_features)
                    table_name = 'hidden'

                    result_url, contents = self.server.db.get_url_query(table_name, y_pred)

                    self.send_obj({'tor': 'hidden', 'client_ip': client_ip, 'url': result_url, 'contents': contents})
                    logger.info('This data is hidden service', result_url, contents)
            else:
                self.send_obj('Error!')
        except Exception as e:
            print('Exception while receiving message: %s' % e)

    def _read(self, size):
        data = b''
        while len(data) < size:
            data_tmp = self.request.recv(size - len(data))
            data += data_tmp
            if data_tmp == b'':
                raise RuntimeError('socket connection broken')

        return data

    def _msg_length(self):
        d = self._read(4)
        s = struct.unpack('!i', d)
        return s[0]

    def read_obj(self):
        size = self._msg_length()
        data = self._read(size)
        frmt = '=%ds' % size
        msg = struct.unpack(frmt, data)
        return json.loads(str(msg[0], 'ascii'))

    def read_file(self):
        msg = self.read_obj()
        msg = json.loads(msg)

        client_ip = msg['client_ip']
        time_synk = msg['time_synk']
        packet_features = msg['packet_features']


        return client_ip, time_synk, packet_features

    def send_obj(self, obj):
        msg = json.dumps(obj)
        frmt = '=%ds' % len(msg)
        packed_msg = struct.pack(frmt, bytes(msg, 'ascii'))
        packed_hdr = struct.pack('!I', len(packed_msg))
        self._sendall(packed_hdr)
        self._sendall(packed_msg)

    def _sendall(self, msg):
        self.request.sendall(msg)


if __name__ == '__main__':

    server = TCPServer(('', 9988), TCPServerHandler)
    server.serve_forever()
