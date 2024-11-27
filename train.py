from Configs import *
from model import *
import pytorch_lightning as pl
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

detection_model=DetectionModel(train_loader=train_loader,val_loader=val_dataloader,
                               lr=lr,batch_size=num_batches)
detection_model.to(device)
trainer = pl.Trainer(max_epochs=epoches,enable_model_summary=True,
                     devices=num_devices,accelerator=device_type,logger=logger,
                     callbacks=[checkpoint_callback,early_stop],enable_progress_bar=True,
                    fast_dev_run=False)
trainer.fit(model=detection_model,
           train_dataloaders=train_loader,
           val_dataloaders=val_dataloader)
trainer.save_checkpoint("detection_model.ckpt")
#detection_model.eval()
#x=torch.rand(1,3,480,640).to(device)
#pred=detection_model(x)
#print(pred)
#for epoch in range(epoches):
#    run=train_one_epoch(model=)