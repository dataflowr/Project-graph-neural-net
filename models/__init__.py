from models.siamese_net import Siamese_Model
from models.similarity_net import Similarity_Model
from models.base_model import Simple_Node_Embedding, BaseModel, Simple_Edge_Embedding


def get_model(args):
    model_instance = _get_model_instance(args["arch"])

    print(f"Fetching model {args['arch']} - {args['model_name']}")
    for i, n in enumerate(args["freeze_mlp"]):
        print(f"{n} frozen MLP in regular block n°{i}")

    model = model_instance(
        original_features_num=args["original_features_num"],
        num_blocks=args["num_blocks"],
        in_features=args["in_features"],
        out_features=args["out_features"],
        depth_of_mlp=args["depth_of_mlp"],
        freeze_mlp=args["freeze_mlp"],
    )
    return model


def _get_model_instance(arch):
    return {
        "Siamese_Model": Siamese_Model,
        "Similarity_Model": Similarity_Model,
        "Simple_Node_Embedding": Simple_Node_Embedding,
        "Simple_Edge_Embedding": Simple_Edge_Embedding,
    }[arch]
