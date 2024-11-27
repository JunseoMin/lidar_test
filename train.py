def train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device, num_epochs=120):
    model.to(device)
    model.train()

    min_loss = min_loss if min_loss else float('inf')   # Initialize the minimum loss

    for epoch in range(start_epoch, num_epochs + 1):
        total_loss = 0
        start_time = time.time()

        for train_data, gt_data in zip(train_dataset, gt_dataset):
            optimizer.zero_grad()
            pred = model(train_data)  # Forward pass
            loss = criterion(pred, gt_data)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Min Loss {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")        
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")
        
        # Save model if loss decreases
        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = "/home/server01/js_ws/lidar_test/ckpt/minloss.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
            print(f"Model saved at {save_path} with loss: {min_loss:.4f}")
            
        # Update learning rate scheduler

        save_path = f"/home/server01/js_ws/lidar_test/ckpt/epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': min_loss
        }, save_path)
        print(f"Model saved at {save_path} with loss: {min_loss:.4f}")

        scheduler.step()
        print(f"==== epoch {epoch} finished ====")

def continue_train_model(model, train_dataset, gt_dataset, optimizer, scheduler, criterion, device,  start_epoch, min_loss = False, num_epochs=120):
    print("========== train start ==========")
    model.to(device)
    model.train()

    min_loss = min_loss if min_loss else float('inf')   # Initialize the minimum loss

    for epoch in range(start_epoch, num_epochs + 1):
        total_loss = 0
        start_time = time.time()

        for train_data, gt_data in zip(train_dataset, gt_dataset):
            optimizer.zero_grad()
            pred = model(train_data)  # Forward pass
            loss = criterion(pred, gt_data)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Min Loss {min_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")        
        print(f"Epoch {epoch} time: {time.time() - start_time:.2f} seconds")
        # Save model if loss decreases
        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = "/home/server01/js_ws/lidar_test/ckpt/new_weights.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'min_loss': min_loss
            }, save_path)
            print(f"Model saved at {save_path} with loss: {min_loss:.4f}")
            
        # Update learning rate scheduler
        scheduler.step()

        save_path = f"/home/server01/js_ws/lidar_test/ckpt/epoch_{epoch}.pth"
        torch.save(model.state_dict(), save_path)

        print(f"==== epoch {epoch} finished ====")