"""ZeroMQ 传输协议消息结构定义与编解码

- ZmqChannelMessageBuffer: channel_write 消息缓冲，编码为紧凑数组 [request_id, topic, [text...]]
- 客户端→服务端载荷（服务端解码）: RequestPayload/CancelPayload/CancelAllPayload/AuditEventPayload/RpcPayload
- 服务端→客户端载荷（客户端解码）: ResponsePayload/RpcResponsePayload/ErrorResponsePayload/CancelAckPayload
- response 载荷中的 value 对象仍按现有 _asdict()/to_dict()/from_dict() 组装解析
"""
from typing import Any, Dict, List, Union, Optional

import msgspec


class ZmqChannelMessageBuffer(msgspec.Struct, array_like=True):
    """按 (request_id, topic) 聚合的 channel 消息缓冲，编码为 [request_id, topic, [text...]]"""
    request_id: str
    topic: str
    text: List[Optional[str]]

    @classmethod
    def new(cls, request_id: str, topic: str, text: Optional[str]) -> 'ZmqChannelMessageBuffer':
        return cls(request_id, topic, [text])

    def add(self, request_id: str, topic: str, text: Optional[str]) -> Optional['ZmqChannelMessageBuffer']:
        """同 key（request_id+topic 相同）原地追加并返回 None；不同 key 返回新建 buffer"""
        if self.request_id == request_id and self.topic == topic:
            self.text.append(text)
            return None
        return ZmqChannelMessageBuffer.new(request_id, topic, text)


# ---------- 客户端→服务端载荷（服务端解码，字段 Optional 带默认，容忍缺失与 null） ----------

class ClientPayload(msgspec.Struct):
    pass


class RequestPayload(ClientPayload):
    client_id: str
    context_id: Optional[str]
    request_id: str
    model_name: str
    messages: List[Dict[str, Any]]
    # compatible with 0.7.0, tools=[]
    tools: Union[Dict[str, Any], List]
    options: Dict[str, Any]
    stream: bool


class CancelPayload(ClientPayload):
    client_id: str
    request_id: str


class CancelAllPayload(ClientPayload):
    client_id: str
    context_id: Optional[str] = None


class AuditEventPayload(ClientPayload):
    client_id: str
    context_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: float


class RpcPayload(ClientPayload):
    request_id: str
    method: str
    kwargs: Dict[str, Any]


CLIENT_PAYLOAD_TYPES = {
    'request': RequestPayload,
    'cancel': CancelPayload,
    'cancel_all': CancelAllPayload,
    'audit_event': AuditEventPayload,
    'rpc': RpcPayload,
}


# ---------- 服务端→客户端载荷（客户端解码） ----------

class ServerPayload(msgspec.Struct):
    pass


class ResponsePayload(ServerPayload):
    request_id: str
    response: Dict[str, Any]


class RpcResponsePayload(ServerPayload):
    request_id: str
    result: Any = None
    error_type: Optional[str] = None
    error_text: Optional[str] = None


class ErrorResponsePayload(ServerPayload):
    type: str
    request_id: str
    error: str


class CancelAckPayload(ServerPayload):
    request_id: str


SERVER_PAYLOAD_TYPES = {
    'response': ResponsePayload,
    'rpc_response': RpcResponsePayload,
    'error': ErrorResponsePayload,
    'cancel_ack': CancelAckPayload,
}


def encode_payload(payload: Any) -> bytes:
    """将消息载荷编码为 JSON bytes"""
    return msgspec.json.encode(payload)


def decode_client_payload(msg_type: str, data: bytes) -> Optional[ClientPayload]:
    payload_type = CLIENT_PAYLOAD_TYPES.get(msg_type)
    if payload_type is None:
        raise ValueError("Unknown request type: " + msg_type)
    return msgspec.json.decode(data, type=payload_type, strict=False)


def decode_server_payload(msg_type: str, data: bytes) -> Optional[ServerPayload]:
    payload_type = SERVER_PAYLOAD_TYPES.get(msg_type)
    if payload_type is None:
        raise ValueError("Unknown response type: " + msg_type)
    return msgspec.json.decode(data, type=payload_type, strict=False)

