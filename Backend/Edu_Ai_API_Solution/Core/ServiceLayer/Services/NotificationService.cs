using DomainLayer.Contracts;
using DomainLayer.Exceptions.Notification;
using DomainLayer.Models;
using Mapster;
using ServiceAbstractionLayer;
using ServiceLayer.Specifications.NotificationSpecifications;
using Shared.Dtos.NotificationDto;

namespace ServiceLayer.Services
{
	public class NotificationService(IUnitOfWork unitOfWork) : INotificationService
	{
		private readonly IUnitOfWork _unitOfWork = unitOfWork;

		public async Task<NotificationBoxResponse> GetUserNotificationsAsync(string userId, CancellationToken cancellationToken = default)
		{
			var box = await GetOrCreateBoxAsync(userId, cancellationToken);

			return new NotificationBoxResponse
			{
				Id = box.Id,
				UnreadCount = box.Notifications.Count(n => !n.IsRead),
				Notifications = box.Notifications
					.OrderBy(n => n.IsRead)                 
					.ThenByDescending(n => n.CreatedAt)      
					.Adapt<IEnumerable<NotificationResponse>>()
			};
		}

		public async Task MarkAsReadAsync(int notificationId, string userId, CancellationToken cancellationToken = default)
		{
			var notificationRepo = _unitOfWork.GetRepository<Notification, int>();

			var spec = new NotificationByIdSpecification(notificationId);
			var notification = await notificationRepo.GetFirstOrDefaultAsync(spec, cancellationToken);

			if (notification is null || notification.NotificationBox.UserId != userId)
				throw new NotificationNotFoundException(notificationId);

			if (notification.IsRead)
				throw new NotificationAlreadyReadException(notificationId);

			notification.IsRead = true;
			notificationRepo.Update(notification);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task MarkAllAsReadAsync(string userId, CancellationToken cancellationToken = default)
		{
			var box = await GetOrCreateBoxAsync(userId, cancellationToken);
			var notificationRepo = _unitOfWork.GetRepository<Notification, int>();

			foreach (var notification in box.Notifications.Where(n => !n.IsRead))
			{
				notification.IsRead = true;
				notificationRepo.Update(notification);
			}

			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task CreateNotificationAsync(string userId, string title, string message, CancellationToken cancellationToken = default)
		{
			var box = await GetOrCreateBoxAsync(userId, cancellationToken);
			var notificationRepo = _unitOfWork.GetRepository<Notification, int>();

			var notification = new Notification
			{
				Title = title,
				Message = message,
				NotificationBoxId = box.Id
			};

			await notificationRepo.AddAsync(notification, cancellationToken);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		public async Task DeleteNotificationAsync(int notificationId, string userId, CancellationToken cancellationToken = default)
		{
			var notificationRepo = _unitOfWork.GetRepository<Notification, int>();

			var spec = new NotificationByIdSpecification(notificationId);
			var notification = await notificationRepo.GetFirstOrDefaultAsync(spec, cancellationToken);

			if (notification is null || notification.NotificationBox.UserId != userId)
				throw new NotificationNotFoundException(notificationId);

			notificationRepo.Delete(notification);
			await _unitOfWork.SaveChangesAsync(cancellationToken);
		}

		private async Task<NotificationBox> GetOrCreateBoxAsync(string userId, CancellationToken cancellationToken)
		{
			var boxRepo = _unitOfWork.GetRepository<NotificationBox, int>();
			var spec = new NotificationBoxByUserSpecification(userId);
			var box = await boxRepo.GetFirstOrDefaultAsync(spec, cancellationToken);

			if (box is null)
			{
				box = new NotificationBox { UserId = userId };
				await boxRepo.AddAsync(box, cancellationToken);
				await _unitOfWork.SaveChangesAsync(cancellationToken);
			}

			return box;
		}
	}
}