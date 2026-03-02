using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.User
{
	public sealed class UserNotFound(string id):NotFoundException($"User With Id : {id} is Not Found")
	{
	}
}
