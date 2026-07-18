from pydantic import BaseModel, Field


class DeviceInfo(BaseModel):
    type: str = Field(description="Hardware attachment type", examples=["pcie", "usb"])
    path: str = Field(description="Device node or sysfs path", examples=["/dev/apex_0"])


class RuntimeStatus(BaseModel):
    interpreter: str | None = Field(description="Python module providing the TFLite interpreter")
    device_preference: str = Field(description="Configured CORALAPI_DEVICE value")
    devices: int = Field(description="Number of Coral Edge TPUs discovered on the host")


class ModelInfo(BaseModel):
    name: str = Field(description="Model name (file stem), used in inference requests")
    path: str = Field(description="Model filename (not the absolute server path)")
    size_bytes: int
    edgetpu_compiled: bool = Field(
        description="True if the file follows the *_edgetpu.tflite naming convention"
    )
    has_labels: bool = Field(description="True if a labels sidecar file was found")
    loaded: bool = Field(description="True if the model is already loaded on the TPU")


class Classification(BaseModel):
    index: int = Field(description="Class index in the model's output tensor")
    label: str | None = Field(description="Human-readable label, if a labels file exists")
    score: float = Field(description="Confidence score")


class ClassifyResponse(BaseModel):
    model: str
    results: list[Classification]


class Detection(BaseModel):
    box: list[float] = Field(
        description="Normalized [ymin, xmin, ymax, xmax] in 0..1, TFLite SSD convention"
    )
    index: int = Field(description="Class index in the model's output tensor")
    label: str | None = Field(description="Human-readable label, if a labels file exists")
    score: float = Field(description="Confidence score")


class DetectResponse(BaseModel):
    model: str
    results: list[Detection]


class RawTensor(BaseModel):
    name: str = Field(description="Output tensor name from the model graph")
    shape: list[int] = Field(description="Tensor shape, e.g. [1, 513, 513, 21]")
    dtype: str = Field(
        description="Element type of `data`: 'float32' when dequantized, else the raw type"
    )
    quantization: dict[str, float] | None = Field(
        description="{'scale', 'zero_point'} if the model output is quantized, else null"
    )
    data: list[float] = Field(
        description="Row-major (C-order) flattened values; reshape with `shape` client-side"
    )


class RawResponse(BaseModel):
    model: str
    input_shape: list[int] = Field(
        description="Shape the uploaded image was resized to before inference (NHWC)"
    )
    outputs: list[RawTensor] = Field(description="Every output tensor of the model, in graph order")


class SegmentClass(BaseModel):
    index: int = Field(description="Class index in the segmentation output")
    label: str | None = Field(description="Human-readable label, if a labels file exists")
    pixels: int = Field(description="Number of pixels assigned to this class")


class SegmentResponse(BaseModel):
    model: str
    width: int = Field(description="Mask width in pixels (the model's output resolution)")
    height: int = Field(description="Mask height in pixels")
    classes: list[SegmentClass] = Field(
        description="Classes present in the mask, most pixels first"
    )
    mask_png: str = Field(
        description="Base64 PNG, mode L, where each pixel value is its class index"
    )


class Keypoint(BaseModel):
    name: str = Field(description="Keypoint name (COCO order for 17-point models)")
    x: float = Field(description="Normalized x in 0..1 (left to right)")
    y: float = Field(description="Normalized y in 0..1 (top to bottom)")
    score: float = Field(description="Keypoint confidence")


class PoseResponse(BaseModel):
    model: str
    keypoints: list[Keypoint]


class EmbedResponse(BaseModel):
    model: str
    dimension: int = Field(description="Length of the embedding vector")
    normalized: bool = Field(description="Whether the vector was L2-normalized")
    embedding: list[float] = Field(description="The feature vector")


class JobStatus(BaseModel):
    id: str = Field(description="Job identifier, returned by the async submit endpoints")
    kind: str = Field(description="Task type", examples=["classify", "detect"])
    model: str
    status: str = Field(
        description="queued | running | succeeded | failed",
        examples=["succeeded"],
    )
    created_at: float = Field(description="Unix timestamp of submission")
    finished_at: float | None = Field(description="Unix timestamp of completion, if finished")
    error: str | None = Field(description="Failure reason when status is 'failed'")
    result: ClassifyResponse | DetectResponse | None = Field(
        description="Inference result when status is 'succeeded'"
    )


class QueueStatus(BaseModel):
    depth: int = Field(description="Configured queue depth (waiting slots)")
    pending: int = Field(description="Requests currently running or waiting for the TPU")
