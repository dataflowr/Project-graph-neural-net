from models.siamese_net import Siamese_Model
from models.similarity_net import Similarity_Model
from models.base_model import Simple_Node_Embedding, BaseModel, Simple_Edge_Embedding


def get_model(args):
    model_instance = _get_model_instance(args["arch"])

    print(
        "Fetching model %s - %s - freeze %i mlp in first block, %i in second, %i in last"
        % (
            args["arch"],
            args["model_name"],
            args["freeze_mlp"][0],
            args["freeze_mlp"][1],
            args["freeze_mlp"][2],
        )
    )
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
