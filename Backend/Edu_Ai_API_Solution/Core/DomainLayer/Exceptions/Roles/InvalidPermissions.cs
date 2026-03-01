using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DomainLayer.Exceptions.Roles
{
	public sealed class InvalidPermissions(): BadRequestException("Invalid Permission")
	{
	}
}
