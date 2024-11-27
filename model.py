import pytorch_lightning as pl
import evaluate
from torch_model import FasterRCNN
import torch
from torchmetrics.detection import MeanAveragePrecision
import torchvision
class DetectionModel(pl.LightningModule):
    def __init__(self,train_loader,val_loader,lr:float,batch_size:int):
        super(DetectionModel,self).__init__()
        self.save_hyperparameters()
        self.model = FasterRCNN(num_classes=2)
        for param in self.model.parameters():
            param.requires_grad_=True
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.map=MeanAveragePrecision()
        self.classification_loss = torch.nn.BCELoss()
        self.objectness_loss = torch.nn.BCELoss()
        self.box_regression_loss = torch.nn.MSELoss()
        self.rpn_regression_box_loss=torch.nn.L1Loss()
        self.train_iou=evaluate.load("mean_iou")
        self.val_iou = evaluate.load("mean_iou")
        self.automatic_optimization = False
    def train_dataloader(self):
        return self.train_loader
    def val_dataloader(self):
        return self.val_loader
    def forward(self,x,target=None):
        pred = self.model(x,target)
        return pred
    def apply_NMS(self,predictions,threshold):
        keep = torchvision.ops.nms(predictions['boxes'], predictions['scores'], threshold)
        final_prediction = {}
        final_prediction['boxes'] = predictions['boxes'][keep]
        final_prediction['scores'] = predictions['scores'][keep]
        final_prediction['labels'] = predictions['labels'][keep]
        return final_prediction
    def training_step(self, batch, batch_idx):
        images, targets = batch[0],batch[1]
        targets_list=[]
        for i in range(len(images)):
            d=dict()
            d["labels"] = targets["labels"][i]
            d["boxes"]=   targets["boxes"][i]
            d["iscrowd"]= targets["iscrowd"][i]
            targets_list.append(d)
        self.model.train()
        predictions = self(x=images, target=targets_list)
        loss_classifier=predictions["loss_classifier"]
        loss_box_reg=predictions["loss_box_reg"]
        loss_objectness=predictions["loss_objectness"]
        loss_rpn_box_reg=predictions["loss_rpn_box_reg"]
        total_loss = loss_objectness+loss_classifier+loss_box_reg+loss_rpn_box_reg
        self.manual_backward(total_loss,retain_graph=True)
        logs = {'train_total_loss': total_loss,
                "train_loss_classifier":loss_classifier,
                "train_loss_box_reg":loss_box_reg,
                "train_loss_rpn_box_reg":loss_rpn_box_reg,
                "train_loss_objectness":loss_objectness}
        for k,v in logs.items():
            self.log(name=k,value=v,on_step=True,on_epoch=True,
                     enable_graph=True,logger=self.logger,
                     batch_size=self.hparams.batch_size,
                     prog_bar=True)

        return logs

    def validation_step(self, batch, batch_idx):
        images, targets = batch[0], batch[1]
        targets_list = []
        for i in range(len(images)):
            d = {
                "labels": targets["labels"][i],
                "boxes": targets["boxes"][i],
                "iscrowd": targets["iscrowd"][i]
            }
            targets_list.append(d)
        self.model.eval()
        predictions = self(images)
        preds_for_map = []
        targets_for_map = []
        for i in range(len(images)):
            pred = {
                "boxes": predictions[i]["boxes"].detach().cpu(),
                "scores": predictions[i]["scores"].detach().cpu(),
                "labels": predictions[i]["labels"].detach().cpu(),
            }
            preds_for_map.append(pred)
            target = {
                "boxes": targets_list[i]["boxes"].detach().cpu(),
                "labels": targets_list[i]["labels"].detach().cpu(),
            }
            targets_for_map.append(target)
        self.map.update(preds_for_map, targets_for_map)
        self.log(name="val_map",value= self.map.compute()["map"],on_step=True, on_epoch=True, prog_bar=True)
        #self.log("val_mean_iou",  on_epoch=True, prog_bar=True)
        self.log_predictions_to_tensorboard(images=images,
                                            true_bounding_boxes=targets,
                                            pred_bounding_boxes=predictions,
                                            mode="validation")
        return {"val_map": self.map.compute()["map"]}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


    def log_predictions_to_tensorboard(self, images, true_bounding_boxes, pred_bounding_boxes, mode: str):
        true_BB_list = []
        pred_BB_list = []
        for i in range(len(pred_bounding_boxes)):
            final_pred=self.apply_NMS(pred_bounding_boxes[i],1)
            pred_BB_list.append(torch.tensor(final_pred["boxes"].clone().detach()))
        for i in range(len(true_bounding_boxes["boxes"])):
            true_BB_list.append(torch.tensor(true_bounding_boxes["boxes"][i].clone().detach()))
        image_with_true_boxes_logs = []
        image_with_pred_boxes_logs = []
        for i in range(len(images)):
            image_with_true_boxes_logs.append(
                torchvision.utils.draw_bounding_boxes(
                    image=images[i],
                    boxes=true_bounding_boxes["boxes"][i],
                    colors="red",
                    width=2
                )
            )
            image_with_pred_boxes_logs.append(
                torchvision.utils.draw_bounding_boxes(
                    image=images[i],
                    boxes=pred_bounding_boxes[i]["boxes"],
                    colors="blue",
                    width=2
                )
            )

        label = torchvision.utils.make_grid(image_with_true_boxes_logs)
        pred = torchvision.utils.make_grid(image_with_pred_boxes_logs)
        self.logger.experiment.add_image(f"{mode}_true_BB", label, self.current_epoch, self.global_step)
        self.logger.experiment.add_image(f"{mode}_pred_BB", pred, self.current_epoch, self.global_step)
