from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



class Trainer(object):
    def __init__(self, step, device, model, dataset_train,
                 dataset_val, criterion, optimizer):
        self.stepped = step
        self.device = device
        self.model = model
        self.dataloader_train = DataLoader(dataset_train,
                                           batch_size=7,
                                           shuffle=True)
        self.dataset_val = dataset_val
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluate = evaluate

    def iterate(self):
        print('Start the training')
        for step, (input, mask, gt) in enumerate(self.dataloader_train):
            torch.cuda.empty_cache()
            loss_dict = self.train(step+self.stepped, input, mask, gt)
            # report the loss

            torch.cuda.empty_cache()

            # save the model
            if (step+self.stepped + 1) % 20 == 0 \
                    or (step + 1) == 200000:


                modelPath = 'drive/MyDrive/model1/modelFile1.pth'
                modelPathtemp = 'drive/MyDrive/model1/modelFiletemp.pth'
                print('Saving the model...')
                '''
                save_ckpt('{}/models/{}.pth'.format('ckpt',
                                                    step+self.stepped + 1),
                          [('model', self.model)],
                          [('optimizer', self.optimizer)],
                          step+self.stepped + 1)
                          '''

                torch.save(
                    {
                        'model':self.model.state_dict(),
                        'optimizer':self.optimizer.state_dict()
                    },
                    modelPath
                )

                torch.save(
                    {
                        'model':self.model.state_dict(),
                        'optimizer':self.optimizer.state_dict()
                    },
                    modelPathtemp
                )

            if step >= 200000:
                break

            print('End of First Iterations...')

    def train(self, step, input, mask, gt):
        torch.cuda.empty_cache()
        # set the model to training mode
        self.model.train()

        # send the input tensors to cuda
        input = input.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        # model forward
        output, _ = self.model(input, mask)


        loss_dict = self.criterion(input, mask, output, gt)
        loss = 0.0
        loss = loss_dict['valid'] +6 * loss_dict['hole'] + 0.05 * loss_dict['perc'] + 120 * loss_dict['style'] + 0.1 * loss_dict['tv']


        # updates the model's params
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_dict['total'] = loss

        plt.imshow(output.cpu().detach()[0].permute(1,2,0))

        plt.show()

        plt.imshow(input.cpu().detach()[0].permute(1,2,0))
        plt.show()
        return (loss_dict)

    def report(self, step, loss_dict):
        print('[STEP: {:>6}] | Valid Loss: {:.6f} | Hole Loss: {:.6f}'\
              '| TV Loss: {:.6f} | Perc Loss: {:.6f}'\
              '| Style Loss: {:.6f} | Total Loss: {:.6f}'.format(
                        step, loss_dict['valid'], loss_dict['hole'],
                        loss_dict['tv'], loss_dict['perc'],
                        loss_dict['style'], loss_dict['total']))

