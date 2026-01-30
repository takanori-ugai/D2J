package jp.live.ugai.d2j

import java.net.Authenticator
import java.net.PasswordAuthentication

/**
 * Represents ProxyAuthenticator.
 * @property user The user.
 * @property password The password.
 */
class ProxyAuthenticator(
    /**
     * The user.
     */
    val user: String,
    /**
     * The password.
     */
    val password: String,
) : Authenticator() {
    /**
     * Executes getPasswordAuthentication.
     */
    override fun getPasswordAuthentication(): PasswordAuthentication = PasswordAuthentication(user, password.toCharArray())
}
