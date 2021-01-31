#include <fmt/format.h>
#include "xtorch.h"
using namespace xtorch;
#include "loadData.h"
#include "GradCheck.h"

int main() {
    // gradCheck();

    const int niter = 1000;
    const int numImgFeatures = 3, depth = 64, imgSize = 64, batchSize = 128, inZ = 100;
    const bool Gbias = false, Dbias = false;

    Sequential netG{ConvTranspose2d{inZ, depth * 8, 4, 1, 0, Gbias},
                    BatchNorm2d{depth * 8},
                    ReLU{.2},
                    ConvTranspose2d{depth * 8, depth * 4, 4, 2, 1, Gbias},
                    BatchNorm2d{depth * 4},
                    ReLU{.2},
                    ConvTranspose2d{depth * 4, depth * 2, 4, 2, 1, Gbias},
                    BatchNorm2d{depth * 2},
                    ReLU{.2},
                    ConvTranspose2d{depth * 2, depth, 4, 2, 1, Gbias},
                    BatchNorm2d{depth},
                    ReLU{.2},
                    ConvTranspose2d{depth, numImgFeatures, 4, 2, 1, Gbias},
                    Tanh{}};
    Sequential netD{Conv2d{numImgFeatures, depth, 4, 2, 1, Dbias},
                    ReLU{.2},
                    Conv2d{depth, depth * 2, 4, 2, 1, Dbias},
                    BatchNorm2d{depth * 2},
                    ReLU{.2},
                    Conv2d{depth * 2, depth * 4, 4, 2, 1, Dbias},
                    BatchNorm2d{depth * 4},
                    ReLU{.2},
                    Conv2d{depth * 4, depth * 8, 4, 2, 1, Dbias},
                    BatchNorm2d{depth * 8},
                    ReLU{.2},
                    Conv2d{depth * 8, 1, 4, 1, 0, Dbias},
                    Flatten{}};

    auto weightInit = [](Module& m) {
        if (m.name() == "Conv2d" or m.name() == "ConvTranspose2d") {
            auto params = m.parameters();
            params[0].setValue(.02 * xt::random::randn<double>(params[0].shape()));
        } else if (m.name() == "BatchNorm2d") {
            auto params = m.parameters();
            params[0].setValue(1. + .02 * xt::random::randn<double>(params[0].shape()));
            params[1].setValue(xt::zeros<double>(params[1].shape()));
        }
    };
    netD.apply(weightInit);
    netG.apply(weightInit);

    auto criterion = BCEWithLogitsLoss{};

    auto fixed_noise = Tensor(xt::random::randn<double>({batchSize, inZ, 1, 1}));
    const int real_label = 1;
    const int fake_label = 0;

    // optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    // optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    auto optimizerD = SGD(netD.parameters(), .01);
    auto optimizerG = SGD(netG.parameters(), .01);

    auto label = Tensor{xt::ones<double>({batchSize})}; // real_label

    auto loader = DataLoader(batchSize);
    for (int epoch = 0; epoch < niter; ++epoch) {
        for (int i = 0; i < loader.size(); ++i) {
            Tensor batch = loader.loadBatch(i);
            // (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            // train with real
            optimizerD.zeroGrad();
            auto logits = netD(batch);
            auto errD_real = criterion(logits, label);
            errD_real.backward();
            auto D_x = xt::mean(logits.sigmoid().getValue())[0];

            // train with fake
            auto noise = Tensor(xt::random::randn<double>({batchSize, inZ, 1, 1}));
            auto fake = netG(noise);
            label.fill(fake_label);
            logits = netD(fake.detach());
            auto errD_fake = criterion(logits, label);
            errD_fake.backward();

            auto D_G_z1 = xt::mean(logits.sigmoid().getValue())[0];
            auto errD = errD_real + errD_fake;
            optimizerD.step();

            // (2) Update G network: maximize log(D(G(z)))
            optimizerG.zeroGrad();
            label.fill(real_label); // # fake labels are real for generator cost
            logits = netD(fake);    // #Replaces leaf node with root node of some other graph
                                    // #updates value and children of leaf
            auto errG = criterion(logits, label);
            errG.backward();
            auto D_G_z2 = xt::mean(logits.sigmoid().getValue())[0];
            optimizerG.step();

            fmt::print("[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}", epoch, niter, i, loader.size(),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2);
            cout << endl;

            dump_img(xt::view(fake.getValue(), 5), "fake1.jpg");
            dump_img(xt::view(netG(fixed_noise).getValue(), 0), "fake2.jpg");
        }
    }
}
