using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Notification
{
	public sealed class NotificationNotFoundException(int notificationId) : NotFoundException($"Notification with ID {notificationId} not found.")
	{
	}
}
