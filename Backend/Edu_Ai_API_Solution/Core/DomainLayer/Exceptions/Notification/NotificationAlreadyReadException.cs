using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Notification
{
	public sealed class NotificationAlreadyReadException(int notificationId) : ConflictException($"Notification with ID {notificationId} is already read.")
	{
	}
}
