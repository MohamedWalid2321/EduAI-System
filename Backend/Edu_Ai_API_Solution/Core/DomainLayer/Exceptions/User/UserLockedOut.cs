using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.User
{
	public sealed class UserLockedOut(string email) : UnAuthorizedException($"User with Email {email} is locked out.")
	{
	}
}
